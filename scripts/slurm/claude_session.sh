#!/bin/bash
#SBATCH --job-name=claude_session
#SBATCH --time=06:00:00
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --account=def-mpederso
#SBATCH --output=scripts/logs/claude_sessions/%x-%j.out

mkdir -p scripts/logs/claude_sessions

# Keep the session alive for 6 hours
sleep infinity
