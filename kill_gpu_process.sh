lsof /dev/nvidia* | awk '{print $2}' | xargs -I {} kill -9 {}

fuser -v /dev/nvidia* |awk '{for(i=1;i<=NF;i++) print "kill -9 " $i;}' | sh

pids=`ps -ef | grep train.py | grep -v grep | awk '{print $2}'`
if [[ "$pids" != "" ]] ; then
    echo $pids
    echo $pids | xargs kill -9
fi
