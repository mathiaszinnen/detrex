#!/bin/bash -l

#SBATCH --job-name=focal_dino
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12000
#SBATCH --gres=gpu:a100:1
#SBATCH -o /home/%u/logs/focaldino-%x-%j-on-%N.out
#SBATCH -e /home/%u/logs/focaldino-%x-%j-on-%N.err
#SBATCH --mail-type=ALL
#Timelimit format: "hours:minutes:seconds" -- max is 24h
#SBATCH --time=00:05:00

TMPDIR=/scratch/zinnen/$SLURM_JOB_ID

mkdir -p ${TMPDIR}
cd ${TMPDIR}

# when possible transfer data from /cluster to /scratch, /scratch are located in SSD and data transfer would be faster.
# Please exchange the --your_name-- with your name

cp -r /net/cluster/zinnen/detectors/detrex .

cd detrex
mkdir -p data/ODOR-v3

time tar xf /net/cluster/shared_dataset/ODOR/odor3.tar -C ./data/ODOR-v3
ln -s $TMPDIR/detrex/data/ODOR-v3/coco-style ./data/ODOR-v3/coco
export DETECTRON2_DATASETS=./data/ODOR-v3/

source /net/cluster/zinnen/miniconda/etc/profile.d/conda.sh
conda activate detrex

echo "starting train"
python tools/train_net.py --config-file projects/dino/configs/odor3_fn_l_lrf_384_fl4_5scale_50ep.py
#python tools/train.py $1 --work-dir /net/cluster/zinnen/mmdetection-workdirs/x101-64_8141041 --resume-from /net/cluster/zinnen/mmdetection-workdirs/x101-64_8141041/epoch_30.pth

echo "done"

#rm -rf $TMPDIR
