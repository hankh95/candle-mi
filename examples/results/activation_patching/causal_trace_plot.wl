(* ::Package:: *)

(* ================================================================== *)
(* Causal Trace Heatmap: Meng et al. (2022) Figure 1e                 *)
(* candle-mi example: activation_patching                              *)
(*                                                                     *)
(* Reads JSON output from --output and plots a layer x position        *)
(* heatmap of recovery percentages. The grid is transposed to match    *)
(* the paper: tokens on Y-axis, layers on X-axis.                     *)
(* ================================================================== *)

(* --- Import JSON output from the Rust example --- *)
(* Change this to the JSON file you want to plot. *)
jsonFile = "gemma-2-2b.json";

raw = Import[
  FileNameJoin[{NotebookDirectory[], jsonFile}],
  "RawJSON"
];

modelId    = raw["model_id"];
tokens     = raw["tokens"];
nLayers    = raw["n_layers"];
seqLen     = raw["seq_len"];
subjectPos = raw["subject_pos"];
grid       = raw["grid"];

Print["Model: ", modelId];
Print["Tokens: ", tokens];
Print["Grid: ", nLayers, " layers x ", seqLen, " positions"];
Print["Subject position: ", subjectPos, " (\"", tokens[[subjectPos + 1]], "\")"];

(* ================================================================== *)
(* PLOT 1: Causal Trace Heatmap (Figure 1e)                           *)
(*                                                                     *)
(* Paper orientation: X = layer, Y = token position (top to bottom).  *)
(* grid is [layer][position], so Transpose gives [position][layer].   *)
(* ================================================================== *)

gridT = Transpose[grid];

(* Token labels for Y-axis (reversed so first token is at top) *)
tokenLabels = tokens;

heatmap = MatrixPlot[
  gridT,
  ColorFunction -> "TemperatureMap",
  PlotLegends -> BarLegend[Automatic, LegendLabel -> "Recovery (%)"],
  FrameLabel -> {"Token", "Layer"},
  FrameTicks -> {
    {Table[{i, tokenLabels[[i]]}, {i, 1, seqLen}], None},
    {Table[{i, i - 1}, {i, 1, nLayers}], None}
  },
  PlotLabel -> Style[
    "Causal Trace (Meng et al. Figure 1e)\n" <> modelId,
    14, Bold
  ],
  ImageSize -> 700,
  AspectRatio -> seqLen / nLayers
];

(* ================================================================== *)
(* PLOT 2: Subject-position recovery curve (layer sweep)               *)
(* ================================================================== *)

subjectRecovery = raw["subject_recovery"];

recoveryCurve = ListLinePlot[
  subjectRecovery,
  PlotRange -> {Automatic, {-5, 105}},
  AxesLabel -> {"Layer", "Recovery (%)"},
  PlotLabel -> Style[
    "Subject-Position Recovery\n" <> modelId <>
    " (pos " <> ToString[subjectPos] <> " = \"" <>
    tokens[[subjectPos + 1]] <> "\")",
    12, Bold
  ],
  PlotStyle -> {Thick, Blue},
  GridLines -> Automatic,
  ImageSize -> 500,
  Epilog -> {Red, PointSize[Large],
    Point[{First@Ordering[subjectRecovery, -1],
           Max[subjectRecovery]}]}
];

(* ================================================================== *)
(* Export PNGs                                                         *)
(* ================================================================== *)

plotDir = FileNameJoin[{NotebookDirectory[], "plots"}];
If[!DirectoryQ[plotDir], CreateDirectory[plotDir]];

prefix = StringReplace[modelId, "/" -> "_"] <> "_";

Export[
  FileNameJoin[{plotDir, prefix <> "causal_trace_heatmap.png"}],
  heatmap, ImageResolution -> 200
];
Export[
  FileNameJoin[{plotDir, prefix <> "subject_recovery.png"}],
  recoveryCurve, ImageResolution -> 200
];

Print["Exported: ", prefix, "causal_trace_heatmap.png"];
Print["Exported: ", prefix, "subject_recovery.png"];

(* Display *)
Column[{
  Style["Causal Trace Analysis", 20, Bold],
  Style[modelId, 14],
  Spacer[20],
  heatmap,
  Spacer[20],
  recoveryCurve
}]
