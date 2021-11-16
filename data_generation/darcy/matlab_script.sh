#!/bin/bash
#SBATCH -t 04:00:00
module load matlab
srun -t 240 matlab -nodisplay -nosplash -nodesktop -nojvm -r demo2
