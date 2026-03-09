(* ================================================================== *)
(* Recurrent Feedback (Anacrousis): Condition Comparison & Couplet    *)
(* Grid from JSON output.                                             *)
(* candle-mi example: recurrent_feedback                              *)
(*                                                                    *)
(* Usage:                                                             *)
(*   1. Run the example with --output to produce JSON files:          *)
(*      cargo run --release --features transformer                    *)
(*        --example recurrent_feedback --                              *)
(*        --output examples/results/recurrent_feedback/prefill.json   *)
(*      cargo run --release --features transformer                    *)
(*        --example recurrent_feedback -- --sustained                  *)
(*        --loop-start 14 --loop-end 15 --strength 1.0                *)
(*        --output examples/results/recurrent_feedback/sustained.json *)
(*   2. Open this .wl in Mathematica and evaluate.                    *)
(*                                                                    *)
(* References:                                                        *)
(*   Taufeeque et al., arXiv:2407.15421 (DRC planning in Sokoban)     *)
(*   Taufeeque et al., arXiv:2506.10138 (mechanistic description of   *)
(*     planning circuits in the Sokoban RNN)                           *)
(*   Lindsey et al., "On the Biology of a Large Language Model", 2025 *)
(*   plip-rs anacrousis branch (28-condition experiment)               *)
(* ================================================================== *)

(* ============================================================ *)
(* LOAD JSON DATA                                                *)
(* ============================================================ *)

resultsDir = FileNameJoin[{
  NotebookDirectory[], "..", "results", "recurrent_feedback"
}];

prefillFile = FileNameJoin[{resultsDir, "prefill.json"}];
sustainedFile = FileNameJoin[{resultsDir, "sustained.json"}];

hasPrefill = FileExistsQ[prefillFile];
hasSustained = FileExistsQ[sustainedFile];

If[hasPrefill,
  prefillData = Import[prefillFile, "RawJSON"],
  Print["WARNING: " <> prefillFile <> " not found — run the example with --output"]
];
If[hasSustained,
  sustainedData = Import[sustainedFile, "RawJSON"],
  Print["WARNING: " <> sustainedFile <> " not found — run the example with --output"]
];

(* ============================================================ *)
(* HELPER: extract per-couplet data from a JSON association      *)
(* ============================================================ *)

(* Extract {id, target, baseline_word, baseline_ok, recurrent_word, recurrent_ok, result}
   from the "couplets" array in a JSON association. *)
extractCouplets[d_] := d["couplets"];

(* Build the grid and word arrays from one or two JSON runs.
   Each JSON has baseline + one recurrent condition.
   We merge them into a unified grid:
     col 1 = baseline (from prefill JSON — both share the same baseline)
     col 2 = prefill recurrent
     col 3 = sustained recurrent *)

(* ============================================================ *)
(* BUILD DATA TABLES                                             *)
(* ============================================================ *)

If[hasPrefill && hasSustained,
  (* --- Both files available: 3-column grid --- *)
  pfCouplets = prefillData["couplets"];
  susCouplets = sustainedData["couplets"];
  nCouplets = Length[pfCouplets];

  modelId = prefillData["model_id"];
  nLayers = prefillData["n_layers"];

  coupletGrid = Table[
    {
      pfCouplets[[i, "id"]],
      pfCouplets[[i, "target"]],
      If[pfCouplets[[i, "baseline_rhymes"]], 1, 0],
      If[pfCouplets[[i, "recurrent_rhymes"]], 1, 0],
      If[susCouplets[[i, "recurrent_rhymes"]], 1, 0]
    },
    {i, nCouplets}
  ];

  generatedWords = Table[
    {
      pfCouplets[[i, "id"]],
      pfCouplets[[i, "baseline_word"]],
      pfCouplets[[i, "recurrent_word"]],
      susCouplets[[i, "recurrent_word"]]
    },
    {i, nCouplets}
  ];

  conditionLabels = {
    "Baseline",
    "Prefill\nL" <> ToString[prefillData["loop_start"]] <>
      "\[Dash]" <> ToString[prefillData["loop_end"]] <>
      ", s=" <> ToString[prefillData["strength"]],
    "Sustained\nL" <> ToString[sustainedData["loop_start"]] <>
      "\[Dash]" <> ToString[sustainedData["loop_end"]] <>
      ", s=" <> ToString[sustainedData["strength"]]
  };

  conditionRhymes = {
    prefillData["baseline_rhymes"],
    prefillData["recurrent_rhymes"],
    sustainedData["recurrent_rhymes"]
  };

  nCols = 3;
  baselineRhymes = prefillData["baseline_rhymes"],

  (* --- Only prefill available: 2-column grid --- *)
  If[hasPrefill && !hasSustained,
    pfCouplets = prefillData["couplets"];
    nCouplets = Length[pfCouplets];
    modelId = prefillData["model_id"];
    nLayers = prefillData["n_layers"];

    coupletGrid = Table[
      {
        pfCouplets[[i, "id"]],
        pfCouplets[[i, "target"]],
        If[pfCouplets[[i, "baseline_rhymes"]], 1, 0],
        If[pfCouplets[[i, "recurrent_rhymes"]], 1, 0]
      },
      {i, nCouplets}
    ];

    generatedWords = Table[
      {
        pfCouplets[[i, "id"]],
        pfCouplets[[i, "baseline_word"]],
        pfCouplets[[i, "recurrent_word"]]
      },
      {i, nCouplets}
    ];

    conditionLabels = {
      "Baseline",
      prefillData["mode"] <> "\nL" <>
        ToString[prefillData["loop_start"]] <>
        "\[Dash]" <> ToString[prefillData["loop_end"]] <>
        ", s=" <> ToString[prefillData["strength"]]
    };

    conditionRhymes = {
      prefillData["baseline_rhymes"],
      prefillData["recurrent_rhymes"]
    };

    nCols = 2;
    baselineRhymes = prefillData["baseline_rhymes"],

    (* --- Neither file: abort --- *)
    Print["ERROR: No JSON files found. Run the example with --output first."];
    Abort[]
  ]
];

(* ============================================================ *)
(* FIGURE 1: Condition Comparison Bar Chart                      *)
(* ============================================================ *)

fig1 = Show[
  BarChart[conditionRhymes,
    ChartLabels -> Placed[
      Style[#, 11] & /@ conditionLabels,
      Below
    ],
    ChartStyle -> Take[
      {GrayLevel[0.5], RGBColor[0.2, 0.4, 0.8], RGBColor[0.85, 0.33, 0.1]},
      nCols
    ],
    Frame -> True,
    FrameLabel -> {
      None,
      Style["Rhyme success (/" <> ToString[nCouplets] <> ")", 14]
    },
    PlotLabel -> Style[
      "Recurrent Feedback: candle-mi (" <> modelId <> ")\n" <>
      "Full-sequence recompute (no KV cache)",
      16, Bold
    ],
    PlotRange -> {Automatic, {0, nCouplets + 0.5}},
    GridLines -> {None, {5, baselineRhymes, 10, nCouplets}},
    GridLinesStyle -> Directive[GrayLevel[0.85]],
    ImageSize -> 700,
    AspectRatio -> 0.6,
    ImagePadding -> {{60, 20}, {80, 50}},
    FrameTicks -> {{Range[0, nCouplets], Automatic}, {None, None}},
    LabelingFunction -> (Placed[#, Above] &)
  ],
  (* Baseline reference line *)
  Graphics[{
    Dashed, GrayLevel[0.5], AbsoluteThickness[1.5],
    Line[{{0, baselineRhymes}, {nCols + 1, baselineRhymes}}],
    Text[
      Style[
        "baseline = " <> ToString[baselineRhymes] <> "/" <> ToString[nCouplets],
        11, Italic, GrayLevel[0.4]
      ],
      {nCols + 0.5, baselineRhymes + 0.4}, {1, -1}
    ]
  }],
  (* Annotate rescued count on best condition *)
  If[Max[conditionRhymes] > baselineRhymes,
    Module[{bestIdx = First[Ordering[conditionRhymes, -1]], rescued},
      rescued = conditionRhymes[[bestIdx]] - baselineRhymes;
      Graphics[{
        RGBColor[0.2, 0.4, 0.8],
        Text[
          Style[
            "+" <> ToString[rescued] <> " rescued",
            12, Bold, RGBColor[0.2, 0.4, 0.8]
          ],
          {bestIdx, conditionRhymes[[bestIdx]] + 0.8}, {0, -1}
        ]
      }]
    ],
    Graphics[{}]
  ]
];

Export[
  FileNameJoin[{NotebookDirectory[], "recurrent_feedback_conditions.png"}],
  fig1, ImageResolution -> 150
];
Print["Exported: recurrent_feedback_conditions.png"];

(* ============================================================ *)
(* FIGURE 2: Per-Couplet Success Grid                            *)
(* ============================================================ *)

successColor = RGBColor[0.3, 0.75, 0.35];  (* Green *)
failureColor = RGBColor[0.9, 0.3, 0.3];    (* Red *)
convertColor = RGBColor[1.0, 0.85, 0.0];   (* Gold -- for conversion highlight *)

fig2grid = Table[
  Module[{val = coupletGrid[[row, col + 2]], word = generatedWords[[row, col + 1]]},
    {If[val == 1, successColor, failureColor],
     Rectangle[{col - 1, nCouplets - row}, {col, nCouplets + 1 - row}],
     Text[
       Style[word, 10, Bold, White],
       {col - 0.5, nCouplets + 0.5 - row}
     ]}
  ],
  {row, nCouplets}, {col, nCols}
];

(* Identify resistant failures: fail under ALL conditions *)
resistantIds = Select[
  Range[nCouplets],
  AllTrue[coupletGrid[[#, 3 ;; nCols + 2]], # == 0 &] &
];
resistantCoupletIds = coupletGrid[[#, 1]] & /@ resistantIds;

(* Row labels: "1. light", "2. play", ... *)
rowLabels = Table[
  Text[
    Style[
      StringJoin[ToString[coupletGrid[[row, 1]]], ". ", coupletGrid[[row, 2]]],
      11,
      If[MemberQ[resistantIds, row], Bold, Plain],
      If[MemberQ[resistantIds, row], RGBColor[1.0, 0.4, 0.4], White]
    ],
    {-0.15, nCouplets + 0.5 - row}, {1, 0}
  ],
  {row, nCouplets}
];

(* Column labels *)
colLabels = Table[
  Text[
    Style[conditionLabels[[col]], 10, Bold, White],
    {col - 0.5, nCouplets + 0.25}, {0, -1}
  ],
  {col, nCols}
];

(* Find rescued couplets: baseline=0, any recurrent=1 *)
rescuedCells = {};
Do[
  Do[
    If[coupletGrid[[row, 3]] == 0 && coupletGrid[[row, col + 2]] == 1,
      AppendTo[rescuedCells, {row, col}]
    ],
    {col, 2, nCols}
  ],
  {row, nCouplets}
];

(* Gold border on rescued couplets *)
rescueHighlights = Table[
  {convertColor, AbsoluteThickness[3],
   Line[{
     {cell[[2]] - 1, nCouplets - cell[[1]]},
     {cell[[2]], nCouplets - cell[[1]]},
     {cell[[2]], nCouplets + 1 - cell[[1]]},
     {cell[[2]] - 1, nCouplets + 1 - cell[[1]]},
     {cell[[2]] - 1, nCouplets - cell[[1]]}
   }]},
  {cell, rescuedCells}
];

fig2 = Graphics[{
  (* Grid cells *)
  Flatten[fig2grid],
  (* Grid lines *)
  GrayLevel[0.95], AbsoluteThickness[1],
  Table[Line[{{0, y}, {nCols, y}}], {y, 0, nCouplets}],
  Table[Line[{{x, 0}, {x, nCouplets}}], {x, 0, nCols}],
  (* Labels *)
  rowLabels,
  colLabels,
  (* Conversion highlights *)
  rescueHighlights
  },
  PlotLabel -> Style[
    "Per-Couplet Results: candle-mi Recurrent Feedback (" <> modelId <> ")",
    14, Bold
  ],
  ImageSize -> 500,
  PlotRange -> {{-2.5, nCols + 0.2}, {-0.5, nCouplets + 1}},
  PlotRangePadding -> {{0, 0.2}, {0.3, 0.5}}
];

Export[
  FileNameJoin[{NotebookDirectory[], "recurrent_feedback_couplet_grid.png"}],
  fig2, ImageResolution -> 150
];
Print["Exported: recurrent_feedback_couplet_grid.png"];

(* ============================================================ *)
(* Display all figures                                           *)
(* ============================================================ *)

Column[{
  Style["Figure 1: Condition Comparison", 14, Bold],
  fig1,
  Spacer[20],
  Style["Figure 2: Per-Couplet Grid", 14, Bold],
  fig2
}]

(* ============================================================ *)
(* Summary                                                       *)
(* ============================================================ *)

Print["\n=== Summary ==="];
Print["Model:          ", modelId];
Print["Conditions:     ", nCols];
Print["Couplets:       ", nCouplets];
Print["Baseline:       ", baselineRhymes, "/", nCouplets];
Print["Best recurrent: ", Max[conditionRhymes], "/", nCouplets];
If[Length[rescuedCells] > 0,
  Print["Rescued cells:  ", Length[rescuedCells]],
  Print["Rescued cells:  0"]
];
If[Length[resistantCoupletIds] > 0,
  Print["Resistant failures: ", resistantCoupletIds]
];
