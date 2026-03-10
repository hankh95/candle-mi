(* ================================================================== *)
(* Character Count Helix: 3D scatter, cosine heatmap, variance bars   *)
(* candle-mi example: character_count_helix                           *)
(* ================================================================== *)

(* --- Import JSON output from the Rust example --- *)
data = Import[
  FileNameJoin[{NotebookDirectory[], "helix_output.json"}],
  "RawJSON"
];

modelId    = data["model_id"];
layer      = data["layer"];
maxCC      = data["max_char_count"];
evr        = data["explained_variance"];
totalVar   = data["total_variance_top6"];
projs      = data["projections"];
cosineSim  = data["cosine_similarity"];

(* Extract projections as {char_count, {pc1, ..., pc6}} *)
charCounts = projs[[All, "char_count"]];
pcCoords   = projs[[All, "pc"]];

(* ================================================================== *)
(* PLOT 1: 3D Helix — PC1 vs PC2 vs PC3, colored by char count       *)
(* ================================================================== *)

helixPoints = MapThread[
  {#1[[1]], #1[[2]], #1[[3]], #2} &,
  {pcCoords, charCounts}
];

helix3D = ListPointPlot3D[
  {#[[1]], #[[2]], #[[3]]} & /@ pcCoords,
  ColorFunction -> (ColorData["Rainbow"][Rescale[#3, {0, 1}]] &),
  PlotStyle -> PointSize[Medium],
  AxesLabel -> {"PC1", "PC2", "PC3"},
  PlotLabel -> Style[
    "Character Count Helix\n" <> modelId <> " layer " <>
    ToString[layer],
    14, Bold
  ],
  ImageSize -> 600,
  Boxed -> True,
  BoxRatios -> {1, 1, 1}
];

(* Add a color bar legend *)
helixWithLegend = Legended[helix3D,
  BarLegend[{"Rainbow", {Min[charCounts], Max[charCounts]}},
    LegendLabel -> "Char count"
  ]
];

(* ================================================================== *)
(* PLOT 2: Cosine Similarity Heatmap                                  *)
(* ================================================================== *)

cosineMatrix = ArrayReshape[cosineSim, {Length[cosineSim], Length[cosineSim]}];

heatmap = MatrixPlot[cosineMatrix,
  ColorFunction -> "TemperatureMap",
  PlotLegends -> Automatic,
  FrameLabel -> {"Char count index", "Char count index"},
  PlotLabel -> Style[
    "Cosine Similarity (ringing pattern)\n" <> modelId <>
    " layer " <> ToString[layer],
    12, Bold
  ],
  ImageSize -> 500
];

(* ================================================================== *)
(* PLOT 3: Explained Variance Bar Chart                               *)
(* ================================================================== *)

varianceBars = BarChart[100 * evr,
  ChartLabels -> Table["PC" <> ToString[i], {i, Length[evr]}],
  PlotLabel -> Style[
    "Explained Variance per Component\nTotal top-" <>
    ToString[Length[evr]] <> ": " <>
    ToString[NumberForm[100 totalVar, {4, 1}]] <> "%",
    12, Bold
  ],
  FrameLabel -> {None, "Variance (%)"},
  Frame -> True,
  ImageSize -> 400,
  ChartStyle -> Lighter[Blue]
];

(* ================================================================== *)
(* PLOT 4: PC4 vs PC5 vs PC6 (secondary twist)                       *)
(* ================================================================== *)

If[Length[First[pcCoords]] >= 6,
  helix456 = ListPointPlot3D[
    {#[[4]], #[[5]], #[[6]]} & /@ pcCoords,
    ColorFunction -> (ColorData["Rainbow"][Rescale[#3, {0, 1}]] &),
    PlotStyle -> PointSize[Medium],
    AxesLabel -> {"PC4", "PC5", "PC6"},
    PlotLabel -> Style[
      "Secondary Twist (PC4-6)\n" <> modelId,
      14, Bold
    ],
    ImageSize -> 600,
    Boxed -> True,
    BoxRatios -> {1, 1, 1}
  ];
  helix456WithLegend = Legended[helix456,
    BarLegend[{"Rainbow", {Min[charCounts], Max[charCounts]}},
      LegendLabel -> "Char count"
    ]
  ];
];

(* ================================================================== *)
(* Export PNGs                                                        *)
(* ================================================================== *)

Export[
  FileNameJoin[{NotebookDirectory[], "helix_pc123.png"}],
  helixWithLegend, ImageResolution -> 200
];
Export[
  FileNameJoin[{NotebookDirectory[], "cosine_heatmap.png"}],
  heatmap, ImageResolution -> 200
];
Export[
  FileNameJoin[{NotebookDirectory[], "variance_bars.png"}],
  varianceBars, ImageResolution -> 200
];
If[ValueQ[helix456WithLegend],
  Export[
    FileNameJoin[{NotebookDirectory[], "helix_pc456.png"}],
    helix456WithLegend, ImageResolution -> 200
  ];
];

Print["Exported: helix_pc123.png, cosine_heatmap.png, variance_bars.png"];
If[ValueQ[helix456WithLegend], Print["Exported: helix_pc456.png"]];

(* ================================================================== *)
(* Display all plots                                                  *)
(* ================================================================== *)

Column[{
  Style["Character Count Helix Analysis", 16, Bold],
  Style[modelId <> " | layer " <> ToString[layer], 12],
  Spacer[10],
  helixWithLegend,
  Spacer[10],
  If[ValueQ[helix456WithLegend], helix456WithLegend, Nothing],
  Spacer[10],
  Row[{varianceBars, Spacer[20], heatmap}]
}]
