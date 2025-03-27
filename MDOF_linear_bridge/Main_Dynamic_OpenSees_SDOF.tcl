# SDOF bilinear

wipe;				                # clear memory of all past model definitions

# Start the model
model BasicBuilder -ndm 1 -ndf 1
node       1        0.00
fix    	   1  		1 
node       10       0.00    -mass 1.00 

 # Following two files are created from Matlab script
source Model_SDOF.tcl;         # SDOF models
source DynAnalysis_SDOF.tcl;         # Earthquake motions

element zeroLength   10      1      10      -mat 10       -dir 1 
element zeroLength   11      1      10      -mat 11       -dir 1 
        

set scale_factor [expr $Factor*9.81]

set tsTag 1
set ptTag 1

timeSeries Path $tsTag -dt $dt -filePath $GMfile -factor $scale_factor	
pattern UniformExcitation $ptTag 1 -accel $tsTag;		# define where and how (pattern tag, dof) 

# Output recorders
#recorder Node    -binary NodeDisp.dat -node $MaxNodeNo -dof 1 disp
source Records_SDOF.tcl;

# Analysis parameters
# ______________________________________________________________________________________________________________________
set gamma 0.5;                      # Newmark integration parameter
set beta  0.25;                     # Newmark integration parameter
set Tol 1.0e-7;                     # convergence tolerance for test

constraints  Transformation
#constraints Penalty 1e10 1e10;     # how it handles boundary conditions
numberer 	Plain;                  # DOF numberer
system      UmfPack;                # how to store and solve the system of equations in the analysis (large model: try UmfPack)
test        NormDispIncr $Tol 100;  # determine if convergence has been achieved at the end of an iteration step
algorithm   ModifiedNewton;                 # use Newton's solution algorithm: updates tangent stiffness at every iteration
integrator  Newmark $gamma $beta;    
analysis    Transient;

# ______________________________________________________________________________________________________________________
#
# Run dynamic analysis
# ______________________________________________________________________________________________________________________
set numSteps [expr int($Nsteps*$dt/$timeInc)]

puts "Running time history analysis..."
analyze $numSteps $timeInc;            
puts "Time history analysis done."

wipe;