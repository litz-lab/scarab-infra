#!/bin/bash

#Read the server's parameters
export SERVER_HEAP_SIZE=14g &&
  export NUM_SERVERS=1

#Prepare Solr
export JAVA_HOME=$(dirname $(dirname $(readlink -f $(which javac))))
export SOLR_JAVA_HOME=$JAVA_HOME
$SOLR_HOME/bin/solr start -force -cloud -p $SOLR_PORT -s $SOLR_CORE_DIR -m $SERVER_HEAP_SIZE
$SOLR_HOME/bin/solr status
$SOLR_HOME/bin/solr create_collection -force -c cloudsuite_web_search -d cloudsuite_web_search -shards $NUM_SERVERS -p $SOLR_PORT

kill -9 $(pgrep java)

# Wait for the process to finish.
while kill -0 $(pgrep java); do
  sleep 1
done 

cd $SOLR_CORE_DIR/cloudsuite_web_search*
rm -rf data
# Copy data from dataset to server
ln -s /download/index_14GB/data data


echo "================================="
echo "Index Node IP Address: "$(hostname -I)
echo "================================="

# /bin/bash

# Run Solr

# cd /home/$username/traces && /home/$username/dynamorio/build/bin64/drrun -t drcachesim -offline -trace_after_instrs 100000000 -exit_after_tracing 101000000 -outdir ./ -- $SOLR_HOME/bin/solr start -force -cloud -f -p $SOLR_PORT -s $SOLR_CORE_DIR -m $SERVER_HEAP_SIZE

# sed -i 's/exec/cd \/home\/$username\/traces \&\& \/home\/$username\/dynamorio\/build\/bin64\/drrun -disable_traces -no_hw_cache_consistency -no_sandbox_writes -no_enable_reset -sandbox2ro_threshold 0 -ro2sandbox_threshold 0 -t drcachesim -offline -trace_after_instrs 100M -exit_after_tracing 101M -outdir .\/ -- /' $SOLR_HOME/bin/solr
# sed -i 's/S -jar start.jar/S -jar \/usr\/src\/solr-9.1.1\/server\/start.jar/' $SOLR_HOME/bin/solr

# $SOLR_HOME/bin/solr start -force -cloud -f -p $SOLR_PORT -s $SOLR_CORE_DIR -m 14g
# echo $SOLR_HOME/bin/solr start -force -cloud -f -p $SOLR_PORT -s $SOLR_CORE_DIR -m $SERVER_HEAP_SIZE

cp -r $SOLR_HOME/server/. /home/$username/traces
cp -r $SOLR_HOME/server/. /home/$username/exp
# (docker run -it --name web_search_client --net host cloudsuite/web-search:client $(hostname -I) 10 && pkill java) &

# (docker run -i --name web_search_client --net host cloudsuite/web-search:client $(hostname -I) 10 && pkill java) &

# java -server -Xms14g -Xmx14g -XX:+UseG1GC -XX:+PerfDisableSharedMem -XX:+ParallelRefProcEnabled -XX:MaxGCPauseMillis=250 -XX:+UseLargePages -XX:+AlwaysPreTouch -XX:+ExplicitGCInvokesConcurrent -Xlog:gc\*:file=/usr/src/solr-9.1.1/server/logs/solr_gc.log:time\,uptime:filecount=9\,filesize=20M -Dsolr.jetty.inetaccess.includes= -Dsolr.jetty.inetaccess.excludes= -DzkClientTimeout=30000 -DzkRun -Dsolr.log.dir=/usr/src/solr-9.1.1/server/logs -Djetty.port=8983 -DSTOP.PORT=7983 -DSTOP.KEY=solrrocks -Duser.timezone=UTC -XX:-OmitStackTraceInFastThrow -XX:OnOutOfMemoryError=/usr/src/solr-9.1.1/bin/oom_solr.sh\ 8983\ /usr/src/solr-9.1.1/server/logs -Djetty.home=/usr/src/solr-9.1.1/server -Dsolr.solr.home=/usr/src/solr_cores -Dsolr.data.home= -Dsolr.install.dir=/usr/src/solr-9.1.1 -Dsolr.default.confdir=/usr/src/solr-9.1.1/server/solr/configsets/_default/conf -Dsolr.jetty.host=0.0.0.0 -Xss256k -XX:CompileCommand=exclude\,com.github.benmanes.caffeine.cache.BoundedLocalCache::put -Djava.security.manager -Djava.security.policy=/usr/src/solr-9.1.1/server/etc/security.policy -Djava.security.properties=/usr/src/solr-9.1.1/server/etc/security.properties -Dsolr.internal.network.permission=\* -DdisableAdminUI=false -jar /usr/src/solr-9.1.1/server/start.jar --module=http --module=requestlog --module=gzip