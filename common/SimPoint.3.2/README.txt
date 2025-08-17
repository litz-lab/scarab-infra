/***********************************************************************
 *  __________________________________________________________________
 *
 *              _____ _           ____        _       __
 *             / ___/(_)___ ___  / __ \____  (_)___  / /_
 *             \__ \/ / __ `__ \/ /_/ / __ \/ / __ \/ __/
 *            ___/ / / / / / / / ____/ /_/ / / / / / /_
 *           /____/_/_/ /_/ /_/_/    \____/_/_/ /_/\__/
 *
 *  __________________________________________________________________
 *
 * This file is part of the SimPoint Toolkit written by Greg Hamerly,
 * Erez Perelman, Jeremy Lau, Tim Sherwood, and Brad Calder as part of
 * Efficient Simulation Project at UCSD.  If you find this toolkit useful please
 * cite the following paper published at ASPLOS 2002.
 *
 *  Timothy Sherwood, Erez Perelman, Greg Hamerly and Brad Calder,
 *  Automatically Characterizing Large Scale Program Behavior , In the
 *  10th International Conference on Architectural Support for Programming
 *  Languages and Operating Systems, October 2002.
 *
 * Contact info:
 *        Brad Calder <calder@cs.ucsd.edu>, (858) 822 - 1619
 *        Greg Hamerly <Greg_Hamerly@baylor.edu>,
 *        Erez Perelman <eperelma@cs.ucsd.edu>,
 *        Jeremy Lau <jl@cs.ucsd.edu>,
 *        Tim Sherwood <sherwood@cs.ucsb.edu>
 *
 *        University of California, San Diego
 *        Department of Computer Science and Engineering
 *        9500 Gilman Drive, Dept 0114
 *        La Jolla CA 92093-0114 USA
 *
 *
 * Copyright 2001, 2002, 2003, 2004, 2005 The Regents of the University of
 * California All Rights Reserved
 *
 * Permission to use, copy, modify and distribute any part of this
 * SimPoint Toolkit for educational, non-profit, and industry research
 * purposes, without fee, and without a written agreement is hereby
 * granted, provided that the above copyright notice, this paragraph and
 * the following four paragraphs appear in all copies and every modified
 * file.
 *
 * Permission is not granted to include SimPoint into a commercial product.
 * Those desiring to incorporate this SimPoint Toolkit into commercial
 * products should contact the Technology Transfer Office, University of
 * California, San Diego, 9500 Gilman Drive, La Jolla, CA 92093-0910, Ph:
 * (619) 534-5815, FAX: (619) 534-7345.
 *
 * IN NO EVENT SHALL THE UNIVERSITY OF CALIFORNIA BE LIABLE TO ANY PARTY
 * FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES,
 * INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THE SimPoint
 * Toolkit, EVEN IF THE UNIVERSITY OF CALIFORNIA HAS BEEN ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * THE SimPoint Toolkit PROVIDED HEREIN IS ON AN "AS IS" BASIS, AND THE
 * UNIVERSITY OF CALIFORNIA HAS NO OBLIGATION TO PROVIDE MAINTENANCE,
 * SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS. THE UNIVERSITY OF
 * CALIFORNIA MAKES NO REPRESENTATIONS AND EXTENDS NO WARRANTIES OF ANY
 * KIND, EITHER IMPLIED OR EXPRESS, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR
 * PURPOSE, OR THAT THE USE OF THE SimPoint Toolkit WILL NOT INFRINGE ANY
 * PATENT, TRADEMARK OR OTHER RIGHTS.
 *
 * No non-profit user may place any restrictions on the use of this
 * software, including as modified by the user, by any other authorized
 * user.
 *
 ************************************************************************/


I. ABOUT SIMPOINT

SimPoint is a simulation analysis tool designed to find the ideal
simulation points in applications. It provides the user with relevant
information regarding the execution behavior of the application,
including an accurate length of the initialization phase.

To make SimPoint a fast and efficient tool, its analysis does not use
any knowledge of the architectural metrics for the program (which
consume a great deal of processing power and time to collect), but
instead utilizes a modeling schema that highly correlates with the
performance of those metrics.

The SimPoint analysis has two main steps. The first step consists of
running a program for an input and recording for each interval of
execution a frequency vector to represent that interval's execution.
The second step analyzes the frequency vector profile and returns the
ideal simulation point, and the duration of initialization phase.
This package includes the software for this second step (the frequency
vector analyzer to find multiple simulation points).  Please see the
following website for packages to generate one form of frequency
vectors called basic block profiles:

http://www-cse.ucsd.edu/~calder/simpoint/


---------------------------------------------------------------------
II. HOW TO USE the SimPoint Toolkit for Simulation

(A) Create a frequency vector profile (e.g. here we will use a basic
block profile, a .bb file) for the program you are interested in
finding simulation points for. You can either using one of our
BBTracker tools, or form your own frequency vector file. The format of
the frequency vector file is described below.  Choosing the interval
length for the frequency vector file is important, since this is
assumed to be the length of a single simulation point in the rest of
the analysis below.  For example, if you set the interval length to be
10 million, then each simulation point is calculated assuming you will
simulate each point for 10 million instructions.

(B) usage: Run "simpoint" (as described below) on the frequency vector
file. This will create the following two files -- .simpoints and
.weights files. Each simulation point in the .simpoints file is in
terms of the number of intervals from the *start* of execution to
reach the start of the simulation point.  The weights are in terms of
the percentage of intervals of excution being represented by each
simulation point.

(C) Now that you have the simulation points, you can simulate each
program for N million instructions at each point in the .simpoints
file, where N million is the interval length.  After simulating each
point, you combine all of the results to get an overall program result
using the weights in the .weights file.


---------------------------------------------------------------------
III. SETUP

In this directory you should find the following subdirectories and files:

README.txt - this file
input/ - contains a sample input file
output/ - the default directory for storing output and working files
analysiscode/ - where the C++ code is stored that performs the analysis
bin/ - contains simpoint executable



---------------------------------------------------------------------
IV. BUILDING

Usage: there are three sub commands for the Makefile in this directory:
        make simpoint - builds the SimPoint program to perform the clustering aanalysis
        make all - generates the SimPoint program and runs it on the sample input
        make clean - clean everything up

The simpoint binary is copied into the bin directory.


---------------------------------------------------------------------
V. FREQUENCY VECTOR FILE FORMAT (USING BASIC BLOCK VECTOR AS AN EXAMPLE)

Running SimPoint requires the frequency vector execution history of
the program and the desired simulation duration. Here we describe the
file format in terms of basic block vectors, but any frequency vectors
can be used as long as they use the same format. An example .bb file
can be found in the input directory.  The basic blocks can be profiled
using the ATOM binary instrumentation tool or simplescalar using other
packages contained within this distribution.  The profiler then
outputs for each interval of instructions (e.g., every 10 millions) a
basic block vector representing the number of times each basic block
was executed during that interval.

The number of intervals, or the number of instructions per interval can be
set to any value and the analysis should handle it cleanly.  Read more
about how basic block profiles are generated in the profile generation
packages, in this file we only concern ourselves with the format of the
file.

The basic block profiler should output a .bb file with the following
format:

   T:BB_X:TimesExecuted :BB_Y:TimesExecuted ... :BB_Z:TimesExecuted
   T:BB_X:TimesExecuted :BB_Y:TimesExecuted ... :BB_Z:TimesExecuted
   T:BB_X:TimesExecuted :BB_Y:TimesExecuted ... :BB_Z:TimesExecuted
    ...
    ...
    ...
   T:BB_X:TimesExecuted :BB_Y:TimesExecuted ... :BB_Z:TimesExecuted
   T:BB_X:TimesExecuted :BB_Y:TimesExecuted ... :BB_Z:TimesExecuted

Each line represents an execution interval of interval-size
instructions executed and each line starts with the literal "T".  The
file then contains a representation of a sparse vector as a list of
dimension,value pairs all separated by colons. Each element has two
fields:

1) BB_X - Represents a particular basic block with a basic block
   identification number. Each basic block in the program is
   represented with its own unique basic block identification number.
   The numbering starts at 1, and represents each dimension in the vector.

2) TimesExecuted - The number of times a basic block has executed
   in that execution interval. This is the basic block size (number
   of instructions in the basic block) times the number of times
   the basic block was executed.  If a basic block has not executed at
   all during an interval, than it does not have an entry for that
   interval. Hence, each line will only correspond to the basic
   blocks executed in a particular interval, usually a sparse
   matrix.

---------------------------------------------------------------------
VI. USAGE Examples


To run SimPoint for computing up to a maximum of 30 simulation points
using binary search for a single seed initialization for each clustering:


Command-line: "simpoint -maxK 30 -numInitSeeds 1 -loadFVFile gcc-00-166-ref"
Using these options (*** indicates user-specified option):
*** -loadFVFile : gcc-00-166-ref
    -k : search
    -iters : 100
    -dim : 15
*** -maxK : 30
*** -numInitSeeds : 1
    -coveragePct : 1
    -bicThreshold : 0.9
    -saveAll : false
    -initkm : samp
    -saveLabels :
    -saveSimpoints :
    -saveSimpointWeights :
    -saveVectorWeights :
    -saveInitCtrs :
    -saveFinalCtrs :
    -saveVectorsTxtFmt :
    -saveVectorsBinFmt :
    -saveProjMatrixTxtFmt :
    -saveProjMatrixBinFmt :
    -loadVectorsTxtFmt :
    -loadVectorsBinFmt :
    -loadProjMatrixTxtFmt :
    -loadProjMatrixBinFmt :
    -loadInitCtrs :
    -loadInitLabels :
    -loadVectorWeights :
    -inputVectorsGzipped : false
    -fixedLength : on
    -numFVs : -1
    -FVDim : -1
    -sampleSize : -1
    -seedkm : 493575226
    -seedproj : 2042712918
    -seedsample : 385089224
    -verbose : 0
-------------------------------------------------------------
  Loading data from frequency vector file 'gcc-00-166-ref' (size: 4692x102038)
  Created random projection matrix (size: 102038x15)
  Loaded and projected frequency vector file
  Applying fixed-length vector weights (uniform weights)
  Searching for best clustering for k <= 30

--------------------------------------------------------------
Run number 1 of at most 7, k = 1
--------------------------------------------------------------
  --------------------------------------------------------------
  Initialization seed trial #1 of 1; initialization seed = 493575226
  --------------------------------------------------------------
  Initialized k-means centers using random sampling: 1 centers
  Number of k-means iterations performed: 2
  BIC score: -13200.2
  Distortion: 5984.8
  Distortions/cluster: 5984.8
  Variance: 1.27581
  Variances/cluster: 1.27581
  The best initialization seed trial was #1

--------------------------------------------------------------
Run number 2 of at most 7, k = 30
--------------------------------------------------------------
  --------------------------------------------------------------
  Initialization seed trial #1 of 1; initialization seed = 493575227
  --------------------------------------------------------------
  Initialized k-means centers using random sampling: 30 centers
  Number of k-means iterations performed: 44
  BIC score: 108582
  Distortion: 119.247
  Distortions/cluster: 9.69634 0.166872 1.3202 1.08809 0.0199032 0.109839 0.0750441 70.8016 1.31071 0.049063 0.157854 0.0486661 0.639056 0.00212212 29.4244 0.386966 0.0185713 0.591622 1.1625 0.00201696 0.0214016 0.302739 0.0924497 0.123345 0.00361603 0.185912 0.0347233 0.047781 0.305531 1.05757
  Variance: 0.0255784
  Variances/cluster: 0.0157664 0.00179433 0.0338514 0.0181348 0.0016586 0.000653804 0.000261478 0.13039 0.00642504 0.000402156 0.00751684 0.000182955 0.00213731 1.02025e-05 0.498719 0.00135303 0.000157384 0.0986037 0.0207589 1.7239e-05 0.0107008 0.0216242 0.000783472 0.00587358 2.80312e-05 0.00338022 0.00024453 0.000645689 0.000883037 0.00581085
  The best initialization seed trial was #1

--------------------------------------------------------------
Run number 3 of at most 7, k = 15
--------------------------------------------------------------
  --------------------------------------------------------------
  Initialization seed trial #1 of 1; initialization seed = 493575228
  --------------------------------------------------------------
  Initialized k-means centers using random sampling: 15 centers
  Number of k-means iterations performed: 25
  BIC score: 91980.9
  Distortion: 213.081
  Distortions/cluster: 0.174846 0.361241 84.9402 0.123516 0.238624 0.0199032 8.85981 0.114226 0.361188 23.9325 45.801 0.0948807 0.0896272 40.5783 7.39123
  Variance: 0.0455593
  Variances/cluster: 0.0102851 0.00138939 0.148757 0.000376573 0.00195593 0.0016586 0.00943537 0.000664104 0.00220237 0.0350916 0.206311 0.000296502 0.00029386 0.414065 0.0158951
  The best initialization seed trial was #1

--------------------------------------------------------------
Run number 4 of at most 7, k = 22
--------------------------------------------------------------
  --------------------------------------------------------------
  Initialization seed trial #1 of 1; initialization seed = 493575229
  --------------------------------------------------------------
  Initialized k-means centers using random sampling: 22 centers
  Number of k-means iterations performed: 29
  BIC score: 98820.8
  Distortion: 165.752
  Distortions/cluster: 5.26562 0.0928175 1.05757 0.00201696 4.03812 0.00212212 0.735767 0.134562 0.591622 0.0857724 0.0909795 0.404499 0.264384 0.320546 0.422794 0.0214016 87.8089 30.9343 10.1335 0.106345 0.0814339 23.157
  Variance: 0.0354929
  Variances/cluster: 0.0516238 0.000909975 0.00581085 1.7239e-05 0.0593842 1.02025e-05 0.00399874 0.00213591 0.0986037 0.00038463 0.000433236 0.00163765 0.00179853 0.000638538 0.0248702 0.0107008 0.169515 0.0448974 0.0164505 0.000770617 0.000329692 0.282402
  The best initialization seed trial was #1

--------------------------------------------------------------
Run number 5 of at most 7, k = 18
--------------------------------------------------------------
  --------------------------------------------------------------
  Initialization seed trial #1 of 1; initialization seed = 493575230
  --------------------------------------------------------------
  Initialized k-means centers using random sampling: 18 centers
  Number of k-means iterations performed: 23
  BIC score: 82019.4
  Distortion: 273.225
  Distortions/cluster: 0.0199032 0.409534 9.66287 1.05757 4.5632 29.4244 1.36881 220.53 0.200089 0.0620945 0.0183801 0.0258969 0.735234 0.0358411 0.0453976 4.67245 0.361241 0.0322296
  Variance: 0.0584564
  Variances/cluster: 0.0016586 0.00116345 0.0157376 0.00581085 0.0518545 0.498719 0.00712919 0.26506 0.00256524 0.00055941 0.000154455 0.000119893 0.00186608 0.000218543 0.000138831 0.0104999 0.00138939 0.000140741
  The best initialization seed trial was #1

--------------------------------------------------------------
Run number 6 of at most 7, k = 20
--------------------------------------------------------------
  --------------------------------------------------------------
  Initialization seed trial #1 of 1; initialization seed = 493575231
  --------------------------------------------------------------
  Initialized k-means centers using random sampling: 20 centers
  Number of k-means iterations performed: 47
  BIC score: 58135.3
  Distortion: 533.34
  Distortions/cluster: 0.703321 0.591622 0.0909795 0.244175 0.171466 1.99354 0.529482 0.690324 3.16467 0.28337 0.0928175 0.0857724 0.0814339 1.49732 5.10592 517.724 0.0214016 0.00201696 0.00212212 0.264384
  Variance: 0.114157
  Variances/cluster: 0.00651223 0.0986037 0.000433236 0.000552433 0.000672415 0.00615291 0.00161921 0.00420929 0.03907 0.00120072 0.000909975 0.00038463 0.000329692 0.0139937 0.0173082 0.483403 0.0107008 1.7239e-05 1.02025e-05 0.00179853
  The best initialization seed trial was #1

--------------------------------------------------------------
Run number 7 of at most 7, k = 21
--------------------------------------------------------------
  --------------------------------------------------------------
  Initialization seed trial #1 of 1; initialization seed = 493575232
  --------------------------------------------------------------
  Initialized k-means centers using random sampling: 21 centers
  Number of k-means iterations performed: 19
  BIC score: 92405
  Distortion: 197.018
  Distortions/cluster: 11.7644 18.7937 0.361241 1.35609 106.485 0.369643 2.03601 0.224835 0.232503 0.273034 1.05757 0.348164 1.59714 3.27793 0.199759 0.809744 0.65763 0.0199032 0.752715 0.0453976 46.3562
  Variance: 0.0421791
  Variances/cluster: 0.158978 0.507939 0.00138939 0.00721326 0.190833 0.00165019 0.0169668 0.00270886 0.000504344 0.00128789 0.00581085 0.00105504 0.0371429 0.00764087 0.000850037 0.00192796 0.00332136 0.0016586 0.0136857 0.000138831 0.207875
  The best initialization seed trial was #1

------------------------------------------------------------------
------------------------------------------------------------------
Post-processing runs
------------------------------------------------------------------
------------------------------------------------------------------
  For the BIC threshold, the best clustering was run 4 (k = 22)
  Post-processing run 4 (k = 22)




**************************************************************************
To run SimPoint for computing up to a maximum of 30 simulation points,
and search thru every value of k:

% simpoint -k 1:30 -loadFVFile gcc-00-166-ref.bb


**************************************************************************
To run SimPoint for computing up to a maximum of 30 simulation points,
and search thru every other value of k:

% simpoint -k 2:2:30 -loadFVFile gcc-00-166-ref.bb


**************************************************************************
To run SimPoint thru a specific set values for k:

% simpoint -k 1,4,5,10,25,30 -loadFVFile gcc-00-166-ref.bb


**************************************************************************
To run SimPoint for a known number of simulation points, the -k option
can be used (e.g. for 30 simulation points):

% simpoint -k 30 -loadFVFile gcc-00-166-ref.bb


**************************************************************************
To run SimPoint for computing up to a maximum of 30 simulation points,
using binary search:

% simpoint -maxK 30 -loadFVFile gcc-00-166-ref.bb
or
% simpoint -maxK 30 -k search -loadFVFile gcc-00-166-ref.bb


**************************************************************************
To run SimPoint for computing up to a maximum of 30 simulation points
and saving essential files as 'simpoints' and 'weights'.

% simpoint -maxK 30 -saveSimpoints simpoints -saveSimpointWeights weights -loadFVFile gcc-00-166-ref.bb


**************************************************************************
To run SimPoint for finding simulation points that cover a percentage
(e.g. 90%) of the execution:

% simpoint -maxK 30 -coveragePct .9 -saveSimpoints simpoints
  -saveSimpointWeights weights -loadFVFile gcc-00-166-ref.bb

100% Coverage           90% Coverage
simpoints weights       simpoints  weights
1885 0    0.0390026 0   2613 1     0.0867155 1
2613 1    0.0833333 1   4469 2     0.157463  2
4469 2    0.151321  2   661  3     0.121978  3
661  3    0.117221  3   1781 4     0.155689  4
1781 4    0.149616  4   1159 5     0.0869372 5
1159 5    0.0835465 5   30   6     0.197827  6
30   6    0.190111  6   1341 7     0.120648  7
1341 7    0.115942  7   2403 8     0.0727434 8
2403 8    0.0699062 8

**************************************************************************
To run SimPoint and sample the frequency vector to use up to a max of
10,000 intervals

% simpoint -maxK 30 -sampleSize 10000 -loadFVFile gcc-00-166-ref.bb


**************************************************************************
To run SimPoint for a specified number of seeds (e.g. for only 1 seed
at each value of k):

% simpoint -maxK 30 -numInitSeeds 1 -loadFVFile gcc-00-166-ref.bb

**************************************************************************
To run SimPoint and initialize k-means centers with furthest-first
algorithm:

% simpoint -maxK 30 -initkm ff -loadFVFile gcc-00-166-ref.bb

**************************************************************************
To run SimPoint for finding a clustering for a specified BIC relative
score (80% of best score, instead of 90%):

% simpoint -maxK 30 -bicThreshold .8 -loadFVFile gcc-00-166-ref.bb


**************************************************************************
To run SimPoint for finding simulation points that cover a percentage
(e.g. 90%) of the execution:

% simpoint -maxK 30 -reportLargestPct .9 -loadFVFile gcc-00-166-ref.bb

**************************************************************************
To run SimPoint and save the projected data of the frequency vectors:

% simpoint -maxK 30 -saveProjData projData -loadFVFile gcc-00-166-ref.bb

**************************************************************************
To run SimPoint loading projected data (Note, 'fixedLength' option
must be specified with this option):

% simpoint -maxK 30 -loadProjData projData -fixedLength on

**************************************************************************
To run SimPoint on variable length intervals:

% simpoint -maxK 30 -fixedLength off -loadFVFile gcc-00-166-ref.bb

**************************************************************************
To run SimPoint and sample the frequency vector to use up to a max of
100,000 intervals

**************************************************************************
% simpoint -maxK 30 -sampleSize 100000 -loadFVFile gcc-00-166-ref.bb

**************************************************************************
To run SimPoint and save all simulation points searched thru

% simpoint -maxK 30 -saveAll -saveSimpoints simpoints -loadFVFile gcc-00-166-ref.bb




---------------------------------------------------------------------
VII. HOW IT WORKS

In order to do a clustering with K-means, you need to know how many
clusters to start with.  An in depth description can be found in:

  Greg Hamerly, Erez Perelman, Jeremy Lau, and Brad Calder,
  SimPoint 3.0: Faster and More Flexible Program Analysis , Workshop on
  Modeling, Benchmarking and Simulation, June 2005

and

  Timothy Sherwood, Erez Perelman, Greg Hamerly and Brad Calder.
  Automatically Characterizing Large Scale Program Behavior, In the
  proceedings of the Tenth International Conference on Architectural Support
  for Programming Languages and Operating Systems (ASPLOS 2002), October
  2002. San Jose, California


---------------------------------------------------------------------
VIII. RELEASE NOTES

Wed Feb  1 11:44:41 PST 2006 (release 3.2)
    - fixed compile bug on 64 bit machines (i.e. AMD64 and PPC/OSX)
    - unrolled inner k-means loop for added performance 
    - added our own random number generator (in Utilities.h), so we get
      consistent random numbers across platforms
    - removed some old code in Datapoint/Dataset classes that are not
      currently being used (e.g. computing early indexes)
    - fixed bug in k-means that would give incorrect answer when 0 iterations
      were chosen

