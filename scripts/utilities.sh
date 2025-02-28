#!/bin/bash

# functions
wait_for () {
  # ref: https://askubuntu.com/questions/674333/how-to-pass-an-array-as-function-argument
  # 1: procedure name
  # 2: task list
  local procedure="$1"
  shift
  local taskPids=("$@")
  echo "${taskPids[@]}"
  echo "wait for all $procedure to finish..."
  # ref: https://stackoverflow.com/a/29535256
  for taskPid in ${taskPids[@]}; do
    echo "wait for $taskPid $procedure process"
    if wait $taskPid; then
      echo "$procedure process $taskPid success"
    else
      echo "$procedure process $taskPid fail"
      exit
    fi
  done
}

report_time () {
  # 1: procedure name
  # 2: start
  # 3: end
  local procedure="$1"
  local start="$2"
  local end="$3"
  local runtime=$((end-start))
  local hours=$((runtime / 3600));
  local minutes=$(( (runtime % 3600) / 60 ));
  local seconds=$(( (runtime % 3600) % 60 ));
  echo "$procedure Runtime: $hours:$minutes:$seconds (hh:mm:ss)"
}

wait_for_non_child () {
  # ref: https://askubuntu.com/questions/674333/how-to-pass-an-array-as-function-argument
  # 1: procedure name
  # 2: task list
  local procedure="$1"
  shift
  local taskPids=("$@")
  echo "${taskPids[@]}"
  echo "wait for all $procedure to finish..."
  # ref: https://stackoverflow.com/a/29535256
  for taskPid in ${taskPids[@]}; do
    echo "wait for $taskPid $procedure process"
    while [ -d "/proc/$taskPid" ]; do
      sleep 10 & wait $!
    done
  done
}

set_app_groupname () {
  # set APP_GROUPNAME
  case $APPNAME in
    cassandra | kafka | tomcat | jme | biojava | luindex)
      echo "DaCapo applications"
      APP_GROUPNAME="dacapo"
      ;;
    chirper | http)
      echo "Renaissance applications"
      APP_GROUPNAME="renaissance"
      ;;
    drupal7 | mediawiki | wordpress)
      echo "HHVM OSS-performance applications"
      APP_GROUPNAME="oss"
      ;;
    compression | hashing | mem | proto | cold_swissmap | hot_swissmap | empirical_driver)
      echo "fleetbench applications"
      APP_GROUPNAME="fleetbench"
      ;;
    verilator)
      echo "verilator"
      APP_GROUPNAME="verilator"
      ;;
    dss)
      echo "dss"
      APP_GROUPNAME="dss"
      ;;
    httpd)
      echo "httpd"
      APP_GROUPNAME="httpd"
      ;;
    solr)
      echo "solr"
      APP_GROUPNAME="solr"
      ;;
    500.perlbench_r | 502.gcc_r | 505.mcf_r | 520.omnetpp_r | 523.xalancbmk_r | 525.x264_r | 531.deepsjeng_r | 541.leela_r | 548.exchange2_r | 557.xz_r | \
    503.bwaves_r | 507.cactuBSSN_r | 508.namd_r | 510.parest_r | 511.povray_r | 519.lbm_r | 521.wrf_r | 526.blender_r | 527.cam4_r | 538.imagick_r | 544.nab_r | 549.fotonik3d_r | 554.roms_r | \
    clang | gcc)
      echo "spec2017"
      APP_GROUPNAME="spec2017"
      ;;
    xgboost)
      echo "xgboost"
      APP_GROUPNAME="xgboost"
      ;;
    dv_insert | dv_update | long_multi_update | simple_insert | simple_multi_update | simple_update | wildcard_index_insert | wildcard_index_query)
      echo "mongo-perf"
      APP_GROUPNAME="mongo-perf"
      ;;
    mysql | postgres)
      echo "sysbench"
      APP_GROUPNAME="sysbench"
      ;;
    memcached)
      echo "memcached"
      APP_GROUPNAME="memcached"
      ;;
    gapbs)
      echo "gapbs"
      APP_GROUPNAME="gapbs"
      ;;
    geekbench)
      echo "geekbench"
      APP_GROUPNAME="geekbench"
      ;;
    llama)
      echo "llama"
      APP_GROUPNAME="llama"
      ;;
    rocksdb)
      echo "rocksdb"
      APP_GROUPNAME="rocksdb"
      ;;
    tailbench)
      echo "tailbench"
      APP_GROUPNAME="tailbench"
      ;;
    taobench | feedsim | django| video_transcode_bench)
      echo "DCPerf"
      APP_GROUPNAME="dcperf"
      ;;
    example)
      echo "example"
      APP_GROUPNAME="example"
      ;;
    allbench)
      echo "allbench"
      APP_GROUPNAME="allbench_traces"
      ;;
    isca2024)
      echo "isca2024"
      APP_GROUPNAME="isca2024_udp"
      ;;
    *)
      APP_GROUPNAME="unknown"
      echo "unknown application"
      ;;
  esac
}

set_app_bincmd () {
  # set BINCMD
  case $APPNAME in
    cassandra)
      #crashes under graal when disabling JIT, default JVM seems to work
      BINCMD="java -Djava.compiler=NONE -jar \$tmpdir/dacapo-23.11-chopin.jar $APPNAME -n 1 -t 1"
      DRIO_ARGS="-trace_after_instrs 500000000000 -exit_after_tracing 50000000000 -dr_ops \"-disable_traces -no_enable_reset -no_sandbox_writes -no_hw_cache_consistency\""
      ;;
    kafka)
      BINCMD="\$tmpdir/graalvm-ce-java11-22.3.1/bin/java -Djava.compiler=NONE -jar \$tmpdir/dacapo-23.11-chopin.jar $APPNAME -n 1 -t 1"
      DRIO_ARGS="-trace_after_instrs 200000000000 -exit_after_tracing 50000000000 -dr_ops \"-no_enable_reset -no_sandbox_writes -no_hw_cache_consistency\""
      ;;
    tomcat)
      BINCMD="java -Djava.compiler=NONE -jar \$tmpdir/dacapo-23.11-chopin.jar $APPNAME -n 1 -t 1"
      DRIO_ARGS="-trace_after_instrs 200000000000 -exit_after_tracing 50000000000 -dr_ops \"-no_enable_reset -no_sandbox_writes -no_hw_cache_consistency\""
      ;;
    jme)
      BINCMD="java -Djava.compiler=NONE -jar \$tmpdir/dacapo-23.11-chopin.jar $APPNAME -n 1 -t 1"
      DRIO_ARGS="-trace_after_instrs 140000000000 -exit_after_tracing 50000000000 -dr_ops \"-disable_traces -no_enable_reset -no_sandbox_writes -no_hw_cache_consistency\""
      ;;
    biojava)
      BINCMD="java -Djava.compiler=NONE -jar \$tmpdir/dacapo-23.11-chopin.jar $APPNAME -n 1 -t 1"
      DRIO_ARGS="-trace_after_instrs 1500000000000 -exit_after_tracing 50000000000 -dr_ops \"-disable_traces -no_enable_reset -no_sandbox_writes -no_hw_cache_consistency\""
      ;;
    luindex)
      BINCMD="java -Djava.compiler=NONE -jar \$tmpdir/dacapo-23.11-chopin.jar $APPNAME -n 1 -t 1"
      DRIO_ARGS="-trace_after_instrs 3000000000000 -exit_after_tracing 50000000000 -dr_ops \"-disable_traces -no_enable_reset -no_sandbox_writes -no_hw_cache_consistency\""
      ;;
    chirper | http)
      BINCMD="java -jar \$tmpdir/renaissance-gpl-0.10.0.jar finagle-$APPNAME -r 10"
      ;;
    drupal7 | mediawiki | wordpress)
      BINCMD="\$HHVM \$tmpdir/oss-performance/perf.php --$APPNAME --hhvm:$(echo \$HHVM)"
      ;;
    compression)
      BINCMD="\$tmpdir/.cache/bazel/_bazel_root/107a4c1ce14e7747be85d98e8915ea0d/execroot/com_google_fleetbench/bazel-out/k8-opt-clang/bin/fleetbench/compression/compression_benchmark"
      ;;
    hashing)
      BINCMD="\$tmpdir/.cache/bazel/_bazel_root/107a4c1ce14e7747be85d98e8915ea0d/execroot/com_google_fleetbench/bazel-out/k8-opt-clang/bin/fleetbench/hashing/hashing_benchmark"
      ;;
    mem)
      BINCMD="\$tmpdir/.cache/bazel/_bazel_root/107a4c1ce14e7747be85d98e8915ea0d/execroot/com_google_fleetbench/bazel-out/k8-opt-clang/bin/fleetbench/libc/mem_benchmark"
      ;;
    proto)
      BINCMD="\$tmpdir/.cache/bazel/_bazel_root/107a4c1ce14e7747be85d98e8915ea0d/execroot/com_google_fleetbench/bazel-out/k8-opt-clang/bin/fleetbench/proto/proto_benchmark"
      ;;
    cold_swissmap | hot_swissmap)
      BINCMD="\$tmpdir/.cache/bazel/_bazel_root/107a4c1ce14e7747be85d98e8915ea0d/execroot/com_google_fleetbench/bazel-out/k8-opt-clang/bin/fleetbench/swissmap/$APPNAME"
      BINCMD+="_benchmark"
      ;;
    empirical_driver)
      BINCMD="\$tmpdir/.cache/bazel/_bazel_root/107a4c1ce14e7747be85d98e8915ea0d/execroot/com_google_fleetbench/bazel-out/k8-opt-clang/bin/fleetbench/tcmalloc/empirical_driver"
      ;;
    verilator)
      BINCMD="\$tmpdir/rocket-chip/emulator/emulator-freechips.rocketchip.system-DefaultConfigN8 +cycle-count \$tmpdir/rocket-chip/emulator/dhrystone-head.riscv"
      ;;
    dss)
      BINCMD="DarwinStreamingServer -d"
      ;;
    httpd)
      BINCMD="/usr/local/apache2/bin/httpd -C 'ServerName 172.17.0.2:80' -X"
      ;;
    solr)
      BINCMD="java -server -Xms14g -Xmx14g -XX:+UseG1GC -XX:+PerfDisableSharedMem -XX:+ParallelRefProcEnabled -XX:MaxGCPauseMillis=250 -XX:+UseLargePages -XX:+AlwaysPreTouch -XX:+ExplicitGCInvokesConcurrent -Xlog:gc\*:file=/usr/src/solr-9.1.1/server/logs/solr_gc.log:time\,uptime:filecount=9\,filesize=20M -Dsolr.jetty.inetaccess.includes= -Dsolr.jetty.inetaccess.excludes= -DzkClientTimeout=30000 -DzkRun -Dsolr.log.dir=/usr/src/solr-9.1.1/server/logs -Djetty.port=8983 -DSTOP.PORT=7983 -DSTOP.KEY=solrrocks -Duser.timezone=UTC -XX:-OmitStackTraceInFastThrow -XX:OnOutOfMemoryError=/usr/src/solr-9.1.1/bin/oom_solr.sh\ 8983\ /usr/src/solr-9.1.1/server/logs -Djetty.home=/usr/src/solr-9.1.1/server -Dsolr.solr.home=/usr/src/solr_cores -Dsolr.data.home= -Dsolr.install.dir=/usr/src/solr-9.1.1 -Dsolr.default.confdir=/usr/src/solr-9.1.1/server/solr/configsets/_default/conf -Dsolr.jetty.host=0.0.0.0 -Xss256k -XX:CompileCommand=exclude\,com.github.benmanes.caffeine.cache.BoundedLocalCache::put -Djava.security.manager -Djava.security.policy=/usr/src/solr-9.1.1/server/etc/security.policy -Djava.security.properties=/usr/src/solr-9.1.1/server/etc/security.properties -Dsolr.internal.network.permission=\* -DdisableAdminUI=false -jar /usr/src/solr-9.1.1/server/start.jar --module=http --module=requestlog --module=gzip"
      ;;
    500.perlbench_r | 502.gcc_r | 505.mcf_r | 520.omnetpp_r | 523.xalancbmk_r | 525.x264_r | 531.deepsjeng_r | 541.leela_r | 548.exchange2_r | 557.xz_r | \
    503.bwaves_r | 507.cactuBSSN_r | 508.namd_r | 510.parest_r | 511.povray_r | 519.lbm_r | 521.wrf_r | 526.blender_r | 527.cam4_r | 538.imagick_r | 544.nab_r | 549.fotonik3d_r | 554.roms_r)
      BINCMD="placeholder"
      ;;
    xgboost)
      BINCMD="python3 \$tmpdir/test-arg.py"
      ;;
    dv_insert | dv_update | long_multi_update | simple_insert | simple_multi_update | simple_update | wildcard_index_insert | wildcard_index_query)
      # command to run the server
      BINCMD="/usr/bin/mongod --config /etc/mongod.conf"
      # command for the workload generator (benchmark) - TODO: automate the server-client run
      CLIENT_BINCMD="python3 \$tmpdir/mongo-perf/benchrun.py -f \$tmpdir/mongo-perf/testcases/$APPNAME.js -t 1"
      ;;
    clang)
      BINCMD="/home/\$username/cpu2017/benchspec/CPU/compile-538-clang.sh 538.imagick_r_train"
      ;;
    gcc)
      BINCMD="/bin/bash /home/\$username/cpu2017/bin/runcpu --config=memtrace --action=build 538.imagick_r"
      ;;
    mysql)
      BINCMD="/usr/sbin/mysqld --basedir=/usr --datadir=/var/lib/mysql --plugin-dir=/usr/lib/mysql/plugin --log-error=/var/log/mysql/error.log --pid-file=53302ceef040.pid"
      CLIENT_BINCMD="sysbench \$tmpdir/sysbench/src/lua/oltp_read_write.lua --mysql-host=127.0.0.1 --mysql-port=3306 --mysql-user=sbtest --db-driver=mysql --tables=10 --table-size=10000 --time=600 run"
      ;;
    postgres)
      BINCMD="/usr/lib/postgresql/12/bin/postgres -D /var/lib/postgresql/12/main -c config_file=/etc/postgresql/12/main/postgresql.conf"
      CLIENT_BINCMD="sysbench \$tmpdir/sysbench/src/lua/oltp_point_select.lua --pgsql-host=127.0.0.1 --pgsql-port=5432 --pgsql-user=sbtest --pgsql-password=password --db-driver=pgsql --tables=10 --table-size=10000 --time=600 run"
      ;;
    gapbs)
      #TODO
      ;;
    geekbench)
      #TODO
      ;;
    llama)
      #TODO
      ;;
    rocksdb)
      ENVVAR="DB_DIR=/tmp/rocksdbtest-1000/ WAL_DIR=/tmp/rocksdbtest-1000/ NUM_KEYS=100000000"
      BINCMD="\$tmpdir/rocksdb/db_bench --benchmarks=readwhilewriting,stats --duration=300 --use_existing_db=1 --sync=1 --level0_file_num_compaction_trigger=4 --level0_slowdown_writes_trigger=20 --level0_stop_writes_trigger=30 --max_background_jobs=16 --max_write_buffer_number=8 --undefok=use_blob_cache,use_shared_block_and_blob_cache,blob_cache_size,blob_cache_numshardbits,prepopulate_blob_cache,multiread_batched,cache_low_pri_pool_ratio,prepopulate_block_cache --db=/tmp/rocksdbtest-1000/ --wal_dir=/tmp/rocksdbtest-1000/ --num=100000000 --key_size=20 --value_size=400 --block_size=8192 --cache_size=17179869184 --cache_numshardbits=6 --compression_max_dict_bytes=0 --compression_ratio=0.5 --compression_type=zstd --bytes_per_sync=1048576 --benchmark_write_rate_limit=0 --write_buffer_size=134217728 --target_file_size_base=134217728 --max_bytes_for_level_base=1073741824 --verify_checksum=1 --delete_obsolete_files_period_micros=62914560 --max_bytes_for_level_multiplier=8 --statistics=0 --stats_per_interval=1 --stats_interval_seconds=60 --report_interval_seconds=1 --histogram=1 --memtablerep=skip_list --bloom_bits=10 --open_files=-1 --subcompactions=1 --compaction_style=0 --num_levels=8 --min_level_to_compress=-1 --level_compaction_dynamic_level_bytes=true --pin_l0_filter_and_index_blocks_in_cache=1 --threads=1 --merge_operator=\"put\" --seed=1705975943 --report_file=/tmp/benchmark_readwhilewriting.t64.log.r.csv 2>&1 | tee -a /tmp/benchmark_readwhilewriting.t64.log"
      ;;
    tailbench)
      #TODO
      ;;
    taobench)
      BINCMD="\$tmpdir/DCPerf/benchpress_cli.py run tao_bench_autoscale"
      ;;
    feedsim)
      BINCMD="\$tmpdir/DCPerf/benchpress_cli.py run feedsim_autoscale"
      ;;
    django)
      BINCMD="\$tmpdir/DCPerf/benchpress_cli.py run django_workload_default"
      ;;
    video_transcode_bench)
      BINCMD="\$tmpdir/DCPerf/benchpress_cli.py run video_transcode_bench_svt"
      ;;
    example)
      BINCMD="/home/$USER/scarab/utils/qsort/test_qsort"
      ;;
    allbench)
      echo "No BINCMD available for allbench"
      ;;
    isca2024)
      echo "No BINCMD available for isca2024 trace runs"
      ;;
    *)
      echo "unknown application"
      ;;
  esac
}