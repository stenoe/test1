echo "Copy input data, unarchive"

cd data/ 
# cp ../testdata/*.tar .
tar -xvf *.tar

ls -all

echo "Go to working directory"

#cd ../script1/
pwd

echo "Setup Display"

export DISPLAY=:99
export QT_X11_NO_MITSHM=1
export GSHOSTNAME=boundless-test

echo "Run the model"

Xvfb :99 -ac -noreset & python ama_maebiastaSKA_T2M_v1.py 

ls -all
