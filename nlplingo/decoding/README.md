
# Current

Will need to install two packages: `flask` and `flask-cors`

May remove `nlplingo_decode.par` in the future because there's a similar one in the package root.

Example call:

`source /home/hqiu/ld100/anaconda/bin/activate && source activate tensorflow-1.5`
`PYTHONPATH=/home/hqiu/SVN_PROJECT_ROOT_LOCAL:/home/hqiu/ld100/nlplingo python nlplingo/decoding/service_mode_basic.py  nlplingo/decoding/nlplingo_decode.par /nfs/raid87/u13/users/jfaschin/emerg_m11_m11_08232018/batch_00000 3001`

# Future

May wrapped a more formal decoding module, which can support service mode and batch mode at the same time.
Finish implementing server_mode_proxy.py mode. So we can decoding on several trigger models at the same time.

# Potential bugs

If the first time, you pass in a document which cause the program to exception, due to the (I'm guessing) computation graph natural, it can be treat as `stuck in the middle of some status`, unless you kill and restart the whole program again, it won't work. Maybe there'e a more elegant way of handling exceptions.
