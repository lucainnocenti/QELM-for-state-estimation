(* ::Package:: *)

BeginPackage["QELM`", {"QM`"}];

(* Usage messages *)
probabilityMatrixFromStatesAndPOVM;
sampleFromProbabilities
sampleFromState;
countSamples;
estimateMeasurementProbsForState;
dirtyProbsMatrixFromStates;
soilProbabilityVector;
randomRank1POVM;

trainQELMForObservableFromStates;
trainQELMfromTargetsAndFrequencies;
trainQELM;
qelmData;


(* More usage messages for other functions go here *)

Begin["`Private`"];


(* Computes the dot product of two matrices. *)
opDot[a_, b_] := Tr @ Dot[ConjugateTranspose @ a, b];


(* Generate a random unitary matrix, drawn from the uniform distribution. *)
(* The method is from Maris Ozols, 'How to generate a random unitary matrix'. *)
(* RandomUnitary[m_] := Orthogonalize[
  Map[#[[1]] + I #[[2]]&, #, {2}]& @ RandomReal[
    NormalDistribution[0, 1], {m, m, 2}
  ]
]; *)

(* Takes a state as a vector and returns its density matrix *)
(* QStateToDensityMatrix[ket_List] := KroneckerProduct[ket, Conjugate @ ket]; *)


(* Take a list of density matrices and a POVM, and return the corresponding matrix of probabilities. *)
(* The returned matrix has dimensions num_outcomes x num_states *)
probabilityMatrixFromStatesAndPOVM[states_, povm_] := Table[
    Chop @ opDot[povmElem, state],
    {povmElem, povm},
    {state, states}
];


(* Takes a probability vector and a number of samples and returns a list of outcomes *)
sampleFromProbabilities[probabilities_, numSamples_Integer] := RandomChoice[
    Rule[probabilities, Range @ Length @ probabilities],
    numSamples
];


(* Takes a state and a POVM, and samples from the probability distribution obtained combining them *)
(* The output is a list of the form `{1, 2, 2, 3, ...}` if the outcome "1" happened, etc *)
(* Here states are to be given as density matrices *)
sampleFromState[state_, povm_, numSamples_Integer] := With[{
        probabilities = Chop@Table[opDot[povmElem, state], {povmElem, povm}]
    },
    (* extract numSamples from the probability distribution obtained combining state and povm *)
    (* RandomChoice[Rule[probabilities, Range @ Length @ probabilities], numSamples] *)
    sampleFromProbabilities[probabilities, numSamples]
];


(* Generate a random rank-1 POVM, with numOutcomes outcomes, acting on a dim-dimensional Hilbert space *)
(* The method is from Maris Ozols, 'How to generate a random rank-1 POVM' (copilot wtf??????). *)
randomRank1POVM[numOutcomes_Integer, dim_Integer] := With[{
        randomIsometry = RandomUnitary[numOutcomes][[All, 1 ;; dim]]
    },
    QStateToDensityMatrix /@ randomIsometry
];

(* take series of samples (eg outcomes) and count occurrance of each; takes into account possible 0 occurances by being nice *)
(* The output is a list of the form `{{1, 5}, {2, 9}, ...}` if the outcome "1" happened 5 times, etc *)
countSamples[samplesList_, maxInt_Integer : None] := With[{
        talliedSamples = Tally @ samplesList
    },
    Table[
        {
            int,
            (* makes sure that MMA doesn't throw a fit when an outcome happened 0 times *)
            Select[talliedSamples, #[[1]] == int &] // If[# != {}, #[[1, 2]], 0] &
        },
        (* maxInt is used to specify the entire set of outcome labels (important if some label never occurs from sampling) *)
        {int, If[maxInt === None, Max @ samplesList, maxInt]}
    ]
];


soilProbabilityVector[probabilities_List, numSamples_Integer] := countSamples[
	sampleFromProbabilities[probabilities, numSamples],
	Length @ probabilities
] // Map @ Last // # / N @ numSamples &;


(* Takes a state and a povm, samples from its prob distribution numSamples times, and returns the observed frequency of each outcome *)
(* Ensures that unobserved outcomes are still accounted for, thus always returning a list of length numSamples  *)
estimateMeasurementProbsForState[state_, povm_, numSamples_Integer] := countSamples[
    sampleFromState[state, povm, numSamples],
    Length @ povm
] // Map @ Last // # / N @ numSamples &;


(* Takes a list of states and simulates measuring each one with the given povm, numSamples times.
    trainingStates : list of states (as density matrices)
    povm : list of povm elements (as matrices)
    numSamples : number of times to sample each state
*)
(* each COLUMN contains the (sampled) probabilities for some state. So this is the matrix <mu, rho> in our notation. *)
dirtyProbsMatrixFromStates[trainingStates_, povm_, numSamples_Integer] := With[{
        numOutcomes = Length @ povm
    },
    (* Transpose @ Table[
        Last /@ (
            countSamples[#, numOutcomes] & @ sampleFromState[state, povm, numSamples]
        ),
       {state, trainingStates}
    ] / numSamples // N *)
    estimateMeasurementProbsForState[#, povm, numSamples] & /@ trainingStates // Transpose
];



(* this takes a list of states, a POVM, and target observables; it \
uses them to compute the dirty probability matrix and from that get W \
*)
trainQELMForObservableFromStates[trainingStates_, targetObservables_, povm_, numSamples_Integer] := With[{
        (* assuming equal statistics at train and test here *)
        dirtyProbsMatrix = dirtyProbsMatrixFromStates[trainingStates, povm, numSamples],
        expvalsMatrix = Chop @ Table[
            opDot[ConjugateTranspose @ observable, state],
            {observable, targetObservables},
            {state, trainingStates}
        ]
    },
    trainQELMfromTargetsAndFrequencies[expvalsMatrix, dirtyProbsMatrix]
    (* this returns W *)
];


(* this takes a matrix and removes all rows after the k-th *)
(* this is used to implement the "keepOnlyKSingularValues" option of trainQELMfromTargetsAndFrequencies *)
removeRowsAfterK[matrix_, k_Integer] := ReplacePart[
    matrix,
    {{i_, j_} /; i > k :> 0}
];
(* this takes a matrix and truncates its singular values after k *)
(* this is used to implement the "keepOnlyKSingularValues" option of trainQELMfromTargetsAndFrequencies *)
truncateSingularValuesAfterK[matrix_, k_Integer] := Dot[
    #[[1]],
    removeRowsAfterK[#[[2]], k],
    ConjugateTranspose@#[[3]]
]& @ SingularValueDecomposition @ matrix;


(* this takes target expectation values and a probability matrix *)
(* uses them to perform the training and get the W *)
(* The pseudoinverse is computed after truncation if specified via options *)
Options[trainQELMfromTargetsAndFrequencies] = {
    "keepOnlyKSingularValues" -> None,
    "ridgeRegression" -> None
};
trainQELMfromTargetsAndFrequencies[labels_, frequencies_, opts : OptionsPattern[] ] := With[
    {
        (* truncate singular values if required *)
        frequenciesModified = If[
            IntegerQ[OptionValue["keepOnlyKSingularValues"] ] && OptionValue["keepOnlyKSingularValues"] >= 1,
            truncateSingularValuesAfterK[frequencies, OptionValue["keepOnlyKSingularValues"] ],
            frequencies
        ]
    },
    If[
        NumericQ[OptionValue["ridgeRegression"] ],
        (* use the ridge regressor if required *)
        (* Echo[#,"llll",MatrixForm]& @  *)
        Dot[
            labels,
            Transpose @ frequenciesModified,
            Inverse @ Plus[
                Dot[frequenciesModified, Transpose @ frequenciesModified],
                OptionValue["ridgeRegression"] * IdentityMatrix[Dimensions[frequenciesModified][[1]] ]
            ]
        ],
        (* otherwise use the standard pseudoinverse *)
        Dot[labels, PseudoInverse @ frequenciesModified]
    ]
];


(* General interface to train QELMs, for various types of information given *)
Options[trainQELM] = {
    "trainingStates" -> None,
    "targetObservables" -> None,
    "POVM" -> None,
    "numSamples" -> None,
    "labels" -> None,
    "frequencies" -> None,
    "returnLabels" -> False,
    "keepOnlyKSingularValues" -> None,
    "ridgeRegression" -> None
};
trainQELM[opts : OptionsPattern[]] := Which[
    (* Case 1: Compute labels from trainingStates and targetObservables if not given *)
    OptionValue["trainingStates"] =!= None && OptionValue["targetObservables"] =!= None && OptionValue["labels"] === None,
    With[
        {
            labels = Chop@Table[
                opDot[observable, state],
                {observable, OptionValue["targetObservables"]},
                {state, OptionValue["trainingStates"]}
            ]
        },
        trainQELM[
            "labels" -> labels,
            Sequence @@ FilterRules[{opts}, Except["labels"] ]
        ]
    ],
    (* Case 2: Compute frequencies if not given *)
    OptionValue["trainingStates"] =!= None && OptionValue["POVM"] =!= None && OptionValue["frequencies"] === None,
    With[
        {
            frequencies = dirtyProbsMatrixFromStates[
                OptionValue["trainingStates"],
                OptionValue["POVM"],
                OptionValue["numSamples"]
            ]
        },
        trainQELM[
            "frequencies" -> frequencies,
            Sequence @@ FilterRules[{opts}, Except["frequencies"] ]
        ]
    ],
    (* Case 3: Training with labels and frequencies *)
    OptionValue["labels"] =!= None && OptionValue["frequencies"] =!= None,
    With[{result = trainQELMfromTargetsAndFrequencies[
        OptionValue["labels"],
        OptionValue["frequencies"],
        "keepOnlyKSingularValues" -> OptionValue["keepOnlyKSingularValues"],
        "ridgeRegression" -> OptionValue["ridgeRegression"]
    ]},
        (* Return also the computed target observables, if asked *)
        If[OptionValue["returnLabels"],
            {result, OptionValue["labels"]},
            result
        ]
    ],
    (* Default case: Invalid arguments *)
    True,
    Message[trainQELM::invalidArgs]
];

(* trainQELM also operates on qelmData structures. *)
(* In this case,  *)
trainQELM[qelmData[data_Association], opts : OptionsPattern[] ] := qelmData @ Append[data, {
    "train" -> Append[data["train"],
        "W" -> trainQELM[
            "labels" -> data["train"]["trueOutcomes"],
            "frequencies" -> data["train"]["counts"],
            "numSamples" -> OptionValue["numSamples"]
        ]
    ]
}];


(* train AND test *)
trainAndTestQELMForObservables[trainingStates_, targetObservables_, povm_, testStates_, numSamples_Integer] := With[{
        wMatrix = trainQELMForObservableFromStates[
            trainingStates, targetObservables, povm, numSamples
        ],
        trueExpvalsMatrix = Chop@Table[
            opDot[obs, state],
            {obs, targetObservables},
            {state, testStates}
        ]
    },
    With[{obtainedExpvalsMatrix = Dot[wMatrix, dirtyProbsMatrixFromStates[testStates, povm, numSamples]]},
        (* finally, compute MSEs for each target observable (it's going to be a numObs x 1 matrix) *)
        Total /@ ((obtainedExpvalsMatrix - trueExpvalsMatrix)^2)
    ]
]


expvalsFromStatesAndObservablesKets[kets_, observables_] := Table[
    Chop @ Dot[ConjugateTranspose @ ket, observable, ket],
    {observable, observables},
    {ket, kets}
];
expvalsFromStatesAndObservablesDms[dms_, observables_] := Table[
    Chop @ Tr @ Dot[dm, observable],
    {observable, observables},
    {dm, dms}
];


(* qelmData contains "train" and "test". Each one of these contains "inputStates" and "counts" *)
(* "inputStates" has dimensions num_states x size_states *)
(* "counts" has dimensions num_outcomes x num_states *)
(* If computed, each of train and test should also have a "trueOutcomes" field *)
qelmData[data_Association]["computeTrueExpvals"] := qelmData @ Append[data, {
    "train" -> Append[data["train"],
        "trueOutcomes" -> If[data["statesAsKets"],
            expvalsFromStatesAndObservablesKets[data["train"]["inputStates"], data["targetObservables"] ],
            expvalsFromStatesAndObservablesDms[data["train"]["inputStates"], data["targetObservables"] ]
        ]
    ],
    "test" -> Append[data["test"],
        "trueOutcomes" -> If[data["statesAsKets"],
            expvalsFromStatesAndObservablesKets[data["test"]["inputStates"], data["targetObservables"] ],
            expvalsFromStatesAndObservablesDms[data["test"]["inputStates"], data["targetObservables"] ]
        ]
    ]
}];


qelmData[data_Association][key_String] := data[key];



End[];  (* End `Private` *)

EndPackage[];
