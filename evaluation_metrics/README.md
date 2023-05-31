# Evaluation metrics
The following cpp files are available to compute the metrics on the predicted labels.
The compiled file is ready to run. 

If the file should be modified, the installation steps to compile the file are the following (tested on windows WSL2)

1) Update the linux packages
```
sudo apt update && sudo apt upgrade
```
2) Install boost libraries
```
sudo apt install libboost-all-dev
```
3) Install gnuplot
```
sudo apt install gnuplot-qt
```
4) Install ghostscript (to generate PDFs)
```
sudo apt install ghostscript
```
5) Install texlive-extra-utils (to crop PDFs)
```
sudo apt install texlive-extra-utils
```
6) Compile file
```
g++ -o evaluate_3D_AP40 evaluate_3D_AP40.cpp
```

To move to the linux directory from windows terminal:
```
cd /mnt/c/your_path
```