// SPDX-License-Identifier: MIT OR Apache-2.0

//! Process and GPU memory reporting.
//!
//! Provides [`MemorySnapshot`] to capture current RAM and VRAM usage,
//! and [`MemoryReport`] to measure deltas between two snapshots.
//!
//! # VRAM measurement strategy
//!
//! VRAM is measured using a two-tier approach:
//!
//! 1. **Primary — NVML** (per-process): Dynamically loads `nvml.dll` (Windows)
//!    or `libnvidia-ml.so.1` (Linux) via `libloading` and calls
//!    `nvmlDeviceGetComputeRunningProcesses` to get true per-process GPU memory.
//! 2. **Fallback — `nvidia-smi`** (device-wide): If NVML cannot be loaded,
//!    spawns `nvidia-smi` as a subprocess. Reports device-wide VRAM; the delta
//!    between two snapshots is accurate on single-user machines.
//!
//! # Platform support
//!
//! | Metric | Windows | Linux |
//! |--------|---------|-------|
//! | RAM (RSS) | `K32GetProcessMemoryInfo` (per-process, exact) | `/proc/self/status` `VmRSS` (per-process, exact) |
//! | VRAM (NVML) | `nvml.dll` (per-process, exact) | `libnvidia-ml.so.1` (per-process, exact) |
//! | VRAM (fallback) | `nvidia-smi` (device-wide) | `nvidia-smi` (device-wide) |
//!
//! # Feature gate
//!
//! This module requires `features = ["memory"]`. The `memory` feature relaxes
//! `#![forbid(unsafe_code)]` to `#![deny(unsafe_code)]` for the Windows FFI
//! call to `K32GetProcessMemoryInfo` and for NVML dynamic symbol loading.
//! On Linux RAM measurement, no unsafe code is used.

use crate::{MIError, Result};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Memory snapshot at a point in time.
///
/// Captures process RAM (resident set size) and optionally GPU VRAM.
/// Use [`MemorySnapshot::now`] to take a measurement, and
/// [`MemoryReport::new`] to compute deltas between two snapshots.
///
/// # Example
///
/// ```no_run
/// use candle_mi::MemorySnapshot;
///
/// let before = MemorySnapshot::now(&candle_core::Device::Cpu)?;
/// // ... load a model ...
/// let after = MemorySnapshot::now(&candle_core::Device::Cpu)?;
/// let report = candle_mi::MemoryReport::new(before, after);
/// println!("RAM delta: {:+.1} MB", report.ram_delta_mb());
/// # Ok::<(), candle_mi::MIError>(())
/// ```
#[derive(Debug, Clone)]
pub struct MemorySnapshot {
    /// Process resident set size (working set on Windows) in bytes.
    pub ram_bytes: u64,
    /// GPU memory used in bytes.
    /// Per-process when measured via NVML, device-wide when via `nvidia-smi` fallback.
    /// `None` if no GPU is present or measurement failed.
    pub vram_bytes: Option<u64>,
    /// Total GPU memory on the active device in bytes.
    /// `None` if no GPU is present or measurement failed.
    pub vram_total_bytes: Option<u64>,
    /// Whether the VRAM measurement is per-process (`true`) or device-wide (`false`).
    /// `None` if no VRAM data is available.
    pub vram_per_process: Option<bool>,
}

/// Memory delta between two snapshots.
///
/// Computed from a `before` and `after` [`MemorySnapshot`].
/// Positive deltas mean memory increased; negative means freed.
#[derive(Debug, Clone)]
pub struct MemoryReport {
    /// Snapshot taken before the operation.
    pub before: MemorySnapshot,
    /// Snapshot taken after the operation.
    pub after: MemorySnapshot,
}

impl MemorySnapshot {
    /// Capture current memory state.
    ///
    /// RAM is always measured (per-process RSS). VRAM is measured only if
    /// `device` is CUDA — first via NVML (per-process), falling back to
    /// `nvidia-smi` (device-wide) if NVML is unavailable.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Memory`] if the RAM query fails (platform API error).
    /// VRAM measurement failures are non-fatal — `vram_bytes` is set to `None`.
    pub fn now(device: &candle_core::Device) -> Result<Self> {
        let ram_bytes = process_rss()?;
        let (vram_bytes, vram_total_bytes, per_process) = if device.is_cuda() {
            gpu_memory_used()
        } else {
            (None, None, None)
        };
        Ok(Self {
            ram_bytes,
            vram_bytes,
            vram_total_bytes,
            vram_per_process: per_process,
        })
    }

    /// Format RAM usage as megabytes.
    #[must_use]
    pub fn ram_mb(&self) -> f64 {
        // CAST: u64 → f64, value is memory in bytes — fits in f64 mantissa
        // for any realistic process size (< 2^53 bytes = 8 PB)
        #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
        let mb = self.ram_bytes as f64 / 1_048_576.0;
        mb
    }

    /// Format VRAM usage as megabytes, if available.
    #[must_use]
    pub fn vram_mb(&self) -> Option<f64> {
        // CAST: u64 → f64, same justification as ram_mb
        #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
        self.vram_bytes.map(|b| b as f64 / 1_048_576.0)
    }
}

impl MemoryReport {
    /// Create a report from two snapshots.
    #[must_use]
    pub const fn new(before: MemorySnapshot, after: MemorySnapshot) -> Self {
        Self { before, after }
    }

    /// RAM delta in megabytes (positive = increased).
    #[must_use]
    pub fn ram_delta_mb(&self) -> f64 {
        self.after.ram_mb() - self.before.ram_mb()
    }

    /// VRAM delta in megabytes (positive = increased).
    /// Returns `None` if either snapshot lacks VRAM data.
    #[must_use]
    pub fn vram_delta_mb(&self) -> Option<f64> {
        match (self.after.vram_mb(), self.before.vram_mb()) {
            (Some(after), Some(before)) => Some(after - before),
            (Some(_) | None, None) | (None, Some(_)) => None,
        }
    }

    /// Print a one-line summary of the delta.
    pub fn print_delta(&self, label: &str) {
        let ram = self.ram_delta_mb();
        print!("  {label}: RAM {ram:+.0} MB");
        if let Some(vram) = self.vram_delta_mb() {
            let qualifier = self.vram_qualifier();
            print!("  |  VRAM {vram:+.0} MB{qualifier}");
        }
        println!();
    }

    /// Print a two-line summary showing before → after for both RAM and VRAM.
    pub fn print_before_after(&self, label: &str) {
        println!(
            "  {label}: RAM {:.0} MB → {:.0} MB ({:+.0} MB)",
            self.before.ram_mb(),
            self.after.ram_mb(),
            self.ram_delta_mb(),
        );
        if let (Some(before), Some(after)) = (self.before.vram_mb(), self.after.vram_mb()) {
            // CAST: u64 → f64, same justification as ram_mb
            #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
            let total = self.after.vram_total_bytes.map_or(String::new(), |t| {
                format!(" / {:.0} MB", t as f64 / 1_048_576.0)
            });
            let qualifier = self.vram_qualifier();
            println!(
                "  {label}: VRAM {before:.0} MB → {after:.0} MB ({:+.0} MB{total}){qualifier}",
                after - before,
            );
        }
    }

    /// Return a short qualifier string indicating VRAM measurement quality.
    const fn vram_qualifier(&self) -> &'static str {
        match self.after.vram_per_process {
            Some(true) => " [per-process]",
            Some(false) => " [device-wide]",
            None => "",
        }
    }
}

// ---------------------------------------------------------------------------
// RAM measurement — per-process RSS
// ---------------------------------------------------------------------------

/// Query the current process's resident set size (RSS) in bytes.
///
/// # Platform
///
/// - **Windows**: `K32GetProcessMemoryInfo` → `WorkingSetSize` (exact, per-process).
/// - **Linux**: `/proc/self/status` → `VmRSS` (exact, per-process, no unsafe).
///
/// # Errors
///
/// Returns [`MIError::Memory`] if the platform API call fails.
fn process_rss() -> Result<u64> {
    #[cfg(target_os = "windows")]
    {
        windows_rss()
    }
    #[cfg(target_os = "linux")]
    {
        linux_rss()
    }
    #[cfg(not(any(target_os = "windows", target_os = "linux")))]
    {
        Err(MIError::Memory(
            "RAM measurement not supported on this platform".into(),
        ))
    }
}

// -- Windows ----------------------------------------------------------------

/// Windows FFI types and functions for `K32GetProcessMemoryInfo`.
#[cfg(target_os = "windows")]
mod win_ffi {
    /// `PROCESS_MEMORY_COUNTERS` structure from the Windows API.
    ///
    /// See: <https://learn.microsoft.com/en-us/windows/win32/api/psapi/ns-psapi-process_memory_counters>
    #[repr(C)]
    pub(super) struct ProcessMemoryCounters {
        /// Size of this structure in bytes.
        pub cb: u32,
        /// Number of page faults.
        pub page_fault_count: u32,
        /// Peak working set size in bytes.
        pub peak_working_set_size: usize,
        /// Current working set size in bytes (= RSS).
        pub working_set_size: usize,
        /// Peak paged pool usage in bytes.
        pub quota_peak_paged_pool_usage: usize,
        /// Current paged pool usage in bytes.
        pub quota_paged_pool_usage: usize,
        /// Peak non-paged pool usage in bytes.
        pub quota_peak_non_paged_pool_usage: usize,
        /// Current non-paged pool usage in bytes.
        pub quota_non_paged_pool_usage: usize,
        /// Current pagefile usage in bytes.
        pub pagefile_usage: usize,
        /// Peak pagefile usage in bytes.
        pub peak_pagefile_usage: usize,
    }

    // SAFETY: These are stable Windows API functions with well-defined ABI.
    // GetCurrentProcess always returns a valid pseudo-handle.
    // K32GetProcessMemoryInfo writes to caller-provided memory of known size.
    #[allow(unsafe_code)]
    unsafe extern "system" {
        /// Returns a pseudo-handle to the current process (always valid, never null).
        pub(super) safe fn GetCurrentProcess() -> isize;

        /// Retrieves memory usage information for the specified process.
        pub(super) unsafe fn K32GetProcessMemoryInfo(
            process: isize,
            ppsmem_counters: *mut ProcessMemoryCounters,
            cb: u32,
        ) -> i32;
    }
}

/// Query RSS on Windows via `K32GetProcessMemoryInfo`.
#[cfg(target_os = "windows")]
#[allow(unsafe_code)]
fn windows_rss() -> Result<u64> {
    let mut counters = win_ffi::ProcessMemoryCounters {
        cb: 0,
        page_fault_count: 0,
        peak_working_set_size: 0,
        working_set_size: 0,
        quota_peak_paged_pool_usage: 0,
        quota_paged_pool_usage: 0,
        quota_peak_non_paged_pool_usage: 0,
        quota_non_paged_pool_usage: 0,
        pagefile_usage: 0,
        peak_pagefile_usage: 0,
    };
    // CAST: usize → u32, struct size is 80 bytes on x64 — fits in u32
    #[allow(clippy::as_conversions, clippy::cast_possible_truncation)]
    let cb = std::mem::size_of::<win_ffi::ProcessMemoryCounters>() as u32;
    counters.cb = cb;

    let handle = win_ffi::GetCurrentProcess();

    // SAFETY: K32GetProcessMemoryInfo writes into the stack-allocated
    // `counters` struct, which is correctly sized (cb field set to struct
    // size). The process handle from GetCurrentProcess is a pseudo-handle
    // that is always valid for the lifetime of the process.
    let ok = unsafe { win_ffi::K32GetProcessMemoryInfo(handle, &raw mut counters, cb) };

    if ok != 0 {
        // CAST: usize → u64, working set size in bytes — always fits
        #[allow(clippy::as_conversions)]
        let rss = counters.working_set_size as u64;
        Ok(rss)
    } else {
        Err(MIError::Memory("K32GetProcessMemoryInfo failed".into()))
    }
}

// -- Linux ------------------------------------------------------------------

/// Query RSS on Linux via `/proc/self/status`.
#[cfg(target_os = "linux")]
fn linux_rss() -> Result<u64> {
    let status = std::fs::read_to_string("/proc/self/status")
        .map_err(|e| MIError::Memory(format!("failed to read /proc/self/status: {e}")))?;

    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            let kb_str = rest.trim().trim_end_matches(" kB").trim();
            let kb: u64 = kb_str.parse().map_err(|e| {
                MIError::Memory(format!("failed to parse VmRSS value '{kb_str}': {e}"))
            })?;
            return Ok(kb * 1024);
        }
    }

    Err(MIError::Memory(
        "VmRSS not found in /proc/self/status".into(),
    ))
}

// ---------------------------------------------------------------------------
// VRAM measurement — NVML primary, nvidia-smi fallback
// ---------------------------------------------------------------------------

/// NVML shared library path (stable across driver versions).
#[cfg(target_os = "linux")]
const NVML_LIB_PATH: &str = "libnvidia-ml.so.1";

/// NVML shared library path (stable across driver versions).
#[cfg(target_os = "windows")]
const NVML_LIB_PATH: &str = "nvml.dll";

/// NVML return code: success.
const NVML_SUCCESS: u32 = 0;

/// NVML return code: buffer too small (need to retry with larger buffer).
const NVML_ERROR_INSUFFICIENT_SIZE: u32 = 7;

/// Maximum number of processes to query from NVML in a single call.
/// 64 is generous — most machines have fewer than 10 GPU processes.
const NVML_MAX_PROCESSES: usize = 64;

/// Per-process GPU memory info returned by NVML.
///
/// Matches the C struct `nvmlProcessInfo_v2_t` from the NVML API.
/// See: <https://docs.nvidia.com/deploy/nvml-api/structnvmlProcessInfo__v2__t.html>
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct NvmlProcessInfo {
    /// Process ID.
    pid: u32,
    /// GPU memory used by this process in bytes.
    used_gpu_memory: u64,
    /// GPU instance ID (MIG). Unused in non-MIG mode.
    gpu_instance_id: u32,
    /// Compute instance ID (MIG). Unused in non-MIG mode.
    compute_instance_id: u32,
}

/// NVML memory info for a device.
///
/// Matches the C struct `nvmlMemory_t` from the NVML API.
/// See: <https://docs.nvidia.com/deploy/nvml-api/structnvmlMemory__t.html>
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct NvmlMemoryInfo {
    /// Total GPU memory in bytes.
    total: u64,
    /// Free GPU memory in bytes.
    free: u64,
    /// Used GPU memory in bytes.
    used: u64,
}

/// Opaque NVML device handle.
type NvmlDevice = *mut std::ffi::c_void;

/// Function signature: `nvmlInit_v2() -> nvmlReturn_t`.
type NvmlInitFn = unsafe extern "C" fn() -> u32;

/// Function signature: `nvmlShutdown() -> nvmlReturn_t`.
type NvmlShutdownFn = unsafe extern "C" fn() -> u32;

/// Function signature: `nvmlDeviceGetHandleByIndex_v2(index, *mut device) -> nvmlReturn_t`.
type NvmlDeviceGetHandleByIndexFn = unsafe extern "C" fn(u32, *mut NvmlDevice) -> u32;

/// Function signature: `nvmlDeviceGetMemoryInfo(device, *mut memory) -> nvmlReturn_t`.
type NvmlDeviceGetMemoryInfoFn = unsafe extern "C" fn(NvmlDevice, *mut NvmlMemoryInfo) -> u32;

/// Function signature:
/// `nvmlDeviceGetComputeRunningProcesses_v3(device, *mut count, *mut infos) -> nvmlReturn_t`.
type NvmlDeviceGetComputeRunningProcessesFn =
    unsafe extern "C" fn(NvmlDevice, *mut u32, *mut NvmlProcessInfo) -> u32;

/// Query GPU memory — NVML primary, `nvidia-smi` fallback.
///
/// Returns `(used_bytes, total_bytes, per_process)`:
/// - `per_process = Some(true)` when NVML per-process query succeeded.
/// - `per_process = Some(false)` when falling back to `nvidia-smi` (device-wide).
/// - All `None` if both methods fail.
fn gpu_memory_used() -> (Option<u64>, Option<u64>, Option<bool>) {
    // Try NVML first (per-process)
    if let Some(result) = nvml_query_process_vram() {
        return result;
    }

    // Fallback to nvidia-smi (device-wide)
    let (used, total) = nvidia_smi_query();
    if used.is_some() {
        (used, total, Some(false))
    } else {
        (None, None, None)
    }
}

/// Attempt to query per-process VRAM via NVML.
///
/// Returns `None` if NVML cannot be loaded or any API call fails,
/// signaling the caller to try the fallback path.
#[allow(unsafe_code)]
fn nvml_query_process_vram() -> Option<(Option<u64>, Option<u64>, Option<bool>)> {
    // SAFETY: libloading::Library::new dynamically loads a shared library.
    // The NVML library is a stable NVIDIA driver component with a well-defined
    // C ABI. We load it, call functions, and unload it within this scope.
    let lib = unsafe { libloading::Library::new(NVML_LIB_PATH) }.ok()?;

    // SAFETY: Loading function symbols from the NVML library. Each symbol
    // name matches the documented NVML C API exactly. The function signatures
    // (type aliases above) match the NVML header definitions.
    let init: libloading::Symbol<'_, NvmlInitFn> = unsafe { lib.get(b"nvmlInit_v2\0") }.ok()?;
    let shutdown: libloading::Symbol<'_, NvmlShutdownFn> =
        unsafe { lib.get(b"nvmlShutdown\0") }.ok()?;
    let get_handle: libloading::Symbol<'_, NvmlDeviceGetHandleByIndexFn> =
        unsafe { lib.get(b"nvmlDeviceGetHandleByIndex_v2\0") }.ok()?;
    let get_memory: libloading::Symbol<'_, NvmlDeviceGetMemoryInfoFn> =
        unsafe { lib.get(b"nvmlDeviceGetMemoryInfo\0") }.ok()?;
    let get_processes: libloading::Symbol<'_, NvmlDeviceGetComputeRunningProcessesFn> =
        unsafe { lib.get(b"nvmlDeviceGetComputeRunningProcesses_v3\0") }.ok()?;

    // Initialize NVML
    // SAFETY: nvmlInit_v2 is safe to call from any thread; it initializes
    // internal NVML state. Returns NVML_SUCCESS (0) on success.
    let ret = unsafe { init() };
    if ret != NVML_SUCCESS {
        return None;
    }

    // Get device handle for GPU 0 (primary GPU)
    let mut device: NvmlDevice = std::ptr::null_mut();
    // SAFETY: nvmlDeviceGetHandleByIndex_v2 writes a valid opaque handle
    // into `device` when it returns NVML_SUCCESS. Index 0 = primary GPU.
    let ret = unsafe { get_handle(0, &raw mut device) };
    if ret != NVML_SUCCESS {
        // SAFETY: nvmlShutdown is always safe after a successful nvmlInit.
        unsafe { shutdown() };
        return None;
    }

    // Get total memory for the device
    let mut mem_info = NvmlMemoryInfo {
        total: 0,
        free: 0,
        used: 0,
    };
    // SAFETY: nvmlDeviceGetMemoryInfo writes into the caller-provided
    // NvmlMemoryInfo struct. The device handle is valid (obtained above).
    let ret = unsafe { get_memory(device, &raw mut mem_info) };
    let total_bytes = if ret == NVML_SUCCESS {
        Some(mem_info.total)
    } else {
        None
    };

    // Get per-process memory
    // CAST: usize → u32, NVML_MAX_PROCESSES is 64 — fits in u32
    #[allow(clippy::as_conversions, clippy::cast_possible_truncation)]
    let mut count = NVML_MAX_PROCESSES as u32;
    let mut infos = [NvmlProcessInfo {
        pid: 0,
        used_gpu_memory: 0,
        gpu_instance_id: 0,
        compute_instance_id: 0,
    }; NVML_MAX_PROCESSES];

    // SAFETY: nvmlDeviceGetComputeRunningProcesses_v3 fills `infos` with
    // up to `count` entries and updates `count` to the actual number written.
    // The buffer is stack-allocated with NVML_MAX_PROCESSES slots, which is
    // sufficient for typical workloads.
    let ret = unsafe { get_processes(device, &raw mut count, infos.as_mut_ptr()) };

    // SAFETY: nvmlShutdown pairs with nvmlInit; always called before return.
    unsafe { shutdown() };

    if ret != NVML_SUCCESS && ret != NVML_ERROR_INSUFFICIENT_SIZE {
        return None;
    }

    // Find our process in the list
    let my_pid = std::process::id();
    // CAST: u32 → usize, count is a small process count — always fits
    #[allow(clippy::as_conversions)]
    let actual_count = count as usize;
    let my_vram = infos
        .get(..actual_count)?
        .iter()
        .find(|info| info.pid == my_pid)
        .map(|info| info.used_gpu_memory);

    // Our PID might not be in the list (no active CUDA context yet?) — return None to trigger fallback
    my_vram.map(|used| (Some(used), total_bytes, Some(true)))
}

/// Query GPU memory via `nvidia-smi` (device-wide fallback).
///
/// Returns `(Some(used_bytes), Some(total_bytes))` on success,
/// or `(None, None)` if `nvidia-smi` is not available or fails.
fn nvidia_smi_query() -> (Option<u64>, Option<u64>) {
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output();

    let output = match output {
        Ok(o) if o.status.success() => o,
        _ => return (None, None),
    };

    // BORROW: explicit String::from_utf8_lossy — nvidia-smi output is ASCII
    let stdout = String::from_utf8_lossy(&output.stdout);
    let line = match stdout.lines().next() {
        Some(l) => l.trim(),
        None => return (None, None),
    };

    // Format: "1234, 16384" (used MiB, total MiB)
    let mut parts = line.split(',');
    let used_str = match parts.next() {
        Some(s) => s.trim(),
        None => return (None, None),
    };
    let total_str = match parts.next() {
        Some(s) => s.trim(),
        None => return (None, None),
    };

    let used_mb: u64 = match used_str.parse() {
        Ok(v) => v,
        Err(_) => return (None, None),
    };
    let total_mb: u64 = match total_str.parse() {
        Ok(v) => v,
        Err(_) => return (None, None),
    };

    // nvidia-smi reports in MiB — convert to bytes
    (Some(used_mb * 1_048_576), Some(total_mb * 1_048_576))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_cpu_has_ram() {
        let snap = MemorySnapshot::now(&candle_core::Device::Cpu).unwrap();
        // Process must be using > 0 bytes of RAM
        assert!(snap.ram_bytes > 0, "RAM should be non-zero");
        // CPU device should not have VRAM
        assert!(snap.vram_bytes.is_none(), "CPU should have no VRAM");
        assert!(
            snap.vram_per_process.is_none(),
            "CPU should have no VRAM qualifier"
        );
    }

    #[test]
    fn report_delta_positive_for_allocation() {
        let before = MemorySnapshot {
            ram_bytes: 100 * 1_048_576, // 100 MB
            vram_bytes: Some(500 * 1_048_576),
            vram_total_bytes: Some(16_384 * 1_048_576),
            vram_per_process: Some(true),
        };
        let after = MemorySnapshot {
            ram_bytes: 200 * 1_048_576, // 200 MB
            vram_bytes: Some(1_000 * 1_048_576),
            vram_total_bytes: Some(16_384 * 1_048_576),
            vram_per_process: Some(true),
        };
        let report = MemoryReport::new(before, after);

        let ram_delta = report.ram_delta_mb();
        assert!(
            (ram_delta - 100.0).abs() < 0.01,
            "RAM delta should be ~100 MB, got {ram_delta}"
        );

        let vram_delta = report.vram_delta_mb().unwrap();
        assert!(
            (vram_delta - 500.0).abs() < 0.01,
            "VRAM delta should be ~500 MB, got {vram_delta}"
        );
    }

    #[test]
    fn report_delta_none_when_no_vram() {
        let before = MemorySnapshot {
            ram_bytes: 100,
            vram_bytes: None,
            vram_total_bytes: None,
            vram_per_process: None,
        };
        let after = MemorySnapshot {
            ram_bytes: 200,
            vram_bytes: None,
            vram_total_bytes: None,
            vram_per_process: None,
        };
        let report = MemoryReport::new(before, after);
        assert!(report.vram_delta_mb().is_none());
    }

    #[test]
    fn ram_mb_conversion() {
        let snap = MemorySnapshot {
            ram_bytes: 1_048_576, // exactly 1 MB
            vram_bytes: None,
            vram_total_bytes: None,
            vram_per_process: None,
        };
        assert!((snap.ram_mb() - 1.0).abs() < 0.001);
    }

    #[test]
    fn vram_qualifier_per_process() {
        let snap = MemorySnapshot {
            ram_bytes: 100,
            vram_bytes: Some(500),
            vram_total_bytes: Some(1000),
            vram_per_process: Some(true),
        };
        let report = MemoryReport::new(snap.clone(), snap);
        assert_eq!(report.vram_qualifier(), " [per-process]");
    }

    #[test]
    fn vram_qualifier_device_wide() {
        let snap = MemorySnapshot {
            ram_bytes: 100,
            vram_bytes: Some(500),
            vram_total_bytes: Some(1000),
            vram_per_process: Some(false),
        };
        let report = MemoryReport::new(snap.clone(), snap);
        assert_eq!(report.vram_qualifier(), " [device-wide]");
    }
}
