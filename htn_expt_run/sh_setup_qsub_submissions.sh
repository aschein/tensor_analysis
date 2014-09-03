#!/bin/bash

#sed 's/0.02/0.04/g' qsub_run_expt_diffGammas_0.001-0.02-0.02_20140829.pbs > qsub_run_expt_diffGammas_0.001-0.04-0.04_20140829.pbs
#sed 's/0.02/0.06/g' qsub_run_expt_diffGammas_0.001-0.02-0.02_20140829.pbs > qsub_run_expt_diffGammas_0.001-0.06-0.06_20140829.pbs
#sed 's/0.02/0.08/g' qsub_run_expt_diffGammas_0.001-0.02-0.02_20140829.pbs > qsub_run_expt_diffGammas_0.001-0.08-0.08_20140829.pbs
#sed 's/0.02/0.09/g' qsub_run_expt_diffGammas_0.001-0.02-0.02_20140829.pbs > qsub_run_expt_diffGammas_0.001-0.09-0.09_20140829.pbs
#sed 's/0.02/0.11/g' qsub_run_expt_diffGammas_0.001-0.02-0.02_20140829.pbs > qsub_run_expt_diffGammas_0.001-0.11-0.11_20140829.pbs
#sed 's/0.02/0.12/g' qsub_run_expt_diffGammas_0.001-0.02-0.02_20140829.pbs > qsub_run_expt_diffGammas_0.001-0.12-0.12_20140829.pbs
#sed 's/0.02/0.13/g' qsub_run_expt_diffGammas_0.001-0.02-0.02_20140829.pbs > qsub_run_expt_diffGammas_0.001-0.13-0.13_20140829.pbs
#sed 's/0.02/0.14/g' qsub_run_expt_diffGammas_0.001-0.02-0.02_20140829.pbs > qsub_run_expt_diffGammas_0.001-0.14-0.14_20140829.pbs
#sed 's/0.02/0.15/g' qsub_run_expt_diffGammas_0.001-0.02-0.02_20140829.pbs > qsub_run_expt_diffGammas_0.001-0.15-0.15_20140829.pbs

qsub -q monkeys -l nodes=2:ppn=4,walltime=15:00:00,mem=8gb -j oe -o qsub_run_expt_diffGammas_0.001-0.02-0.02_20140829.out -m abe -M rchen87@gatech.edu qsub_run_expt_diffGammas_0.001-0.02-0.02_20140829.pbs
qsub -q monkeys -l nodes=2:ppn=4,walltime=15:00:00,mem=8gb -j oe -o qsub_run_expt_diffGammas_0.001-0.04-0.04_20140829.out -m abe -M rchen87@gatech.edu qsub_run_expt_diffGammas_0.001-0.04-0.04_20140829.pbs
qsub -q monkeys -l nodes=2:ppn=4,walltime=15:00:00,mem=8gb -j oe -o qsub_run_expt_diffGammas_0.001-0.06-0.06_20140829.out -m abe -M rchen87@gatech.edu qsub_run_expt_diffGammas_0.001-0.06-0.06_20140829.pbs
qsub -q monkeys -l nodes=2:ppn=4,walltime=15:00:00,mem=8gb -j oe -o qsub_run_expt_diffGammas_0.001-0.08-0.08_20140829.out -m abe -M rchen87@gatech.edu qsub_run_expt_diffGammas_0.001-0.08-0.08_20140829.pbs
qsub -q monkeys -l nodes=2:ppn=4,walltime=15:00:00,mem=8gb -j oe -o qsub_run_expt_diffGammas_0.001-0.09-0.09_20140829.out -m abe -M rchen87@gatech.edu qsub_run_expt_diffGammas_0.001-0.09-0.09_20140829.pbs

qsub -q monkeys -l nodes=2:ppn=4,walltime=15:00:00,mem=8gb -j oe -o qsub_run_expt_diffGammas_0.001-0.11-0.11_20140829.out -m abe -M rchen87@gatech.edu qsub_run_expt_diffGammas_0.001-0.11-0.11_20140829.pbs
qsub -q monkeys -l nodes=2:ppn=4,walltime=15:00:00,mem=8gb -j oe -o qsub_run_expt_diffGammas_0.001-0.12-0.12_20140829.out -m abe -M rchen87@gatech.edu qsub_run_expt_diffGammas_0.001-0.12-0.12_20140829.pbs
qsub -q monkeys -l nodes=2:ppn=4,walltime=15:00:00,mem=8gb -j oe -o qsub_run_expt_diffGammas_0.001-0.13-0.13_20140829.out -m abe -M rchen87@gatech.edu qsub_run_expt_diffGammas_0.001-0.13-0.13_20140829.pbs
qsub -q monkeys -l nodes=2:ppn=4,walltime=15:00:00,mem=8gb -j oe -o qsub_run_expt_diffGammas_0.001-0.14-0.14_20140829.out -m abe -M rchen87@gatech.edu qsub_run_expt_diffGammas_0.001-0.14-0.14_20140829.pbs
qsub -q monkeys -l nodes=2:ppn=4,walltime=15:00:00,mem=8gb -j oe -o qsub_run_expt_diffGammas_0.001-0.15-0.15_20140829.out -m abe -M rchen87@gatech.edu qsub_run_expt_diffGammas_0.001-0.15-0.15_20140829.pbs
