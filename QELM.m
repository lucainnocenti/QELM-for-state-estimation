(* ::Package:: *)

BeginPackage["QELM`"];

(* Usage messages *)
sampleFromState;
countSamples;
estimateMeasurementProbsForState;
dirtyProbsMatrixFromStates;


(* More usage messages for other functions go here *)

Begin["`Private`"];


opDot[a_, b_] := Tr @ Dot[ConjugateTranspose @ a, b];


(* Generate a random unitary matrix, drawn from the uniform distribution. *)
(* The method is from Maris Ozols, 'How to generate a random unitary matrix'. *)
RandomUnitary[m_] := Orthogonalize[
  Map[#[[1]] + I #[[2]]&, #, {2}]& @ RandomReal[
    NormalDistribution[0, 1], {m, m, 2}
  ]
];

(* Takes a state as a vector and returns its density matrix *)
QStateToDensityMatrix[ket_List] := KroneckerProduct[ket, Conjugate @ ket];


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
    Dot[
        expvalsMatrix,
        PseudoInverse @ dirtyProbsMatrix
    ] (* this returns W *)
];


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


(* More functions and shit go here *)

End[];  (* End `Private` *)

EndPackage[];
