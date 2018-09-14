#PBS -S /bin/bash
#PBS -q gpu_q
#PBS -l nodes=1:ppn=2:gpus=1
#PBS -l walltime=2:00:00:00
#PBS -l mem=6gb

TRAINING_FOLDER="Eval_Runs"
DATASET_FOLDER="$(pwd)/data"

#Extend Maximum number of directories to recursively search for a Pipfile.
export PIPENV_MAX_DEPTH=10

###
# When the job starts, the script copies all the python files and its' self
# to the folder Training_Runs/$RUN.
# It then cds into the directory and starts Train.py
###

# If a name is given, run locally, otherwise check to see if
# running as a cluster job, If so, act accordingly.
if [ "$#" -eq 1 ]; then
  RUN=$1
elif [ -n "$PBS_JOBNAME" ]; then
  RUN=$PBS_JOBNAME
  cd $PBS_O_WORKDIR
else
  echo "Please provide the run name: run.sh RUN"
  exit
fi

ml CUDA/9.0.176
ml cuDNN/7.0.4-CUDA-9.0.176
ml Python/3.6.4-foss-2018a

mkdir -p $DATASET_FOLDER
mkdir -p $TRAINING_FOLDER/$RUN;
#cp -r nets/ $TRAINING_FOLDER/$RUN/
#cp -r dataloaders/ $TRAINING_FOLDER/$RUN/
#cp -r analysis/ $TRAINING_FOLDER/$RUN/
cp *.py $TRAINING_FOLDER/$RUN/
cp run2.sh $TRAINING_FOLDER/$RUN/

cd $TRAINING_FOLDER/$RUN/

pipenv run python Train.py 2 --Nnoise 0 --download --Nimages 10\
                    --dataset-loc $DATASET_FOLDER --dataset MNIST
