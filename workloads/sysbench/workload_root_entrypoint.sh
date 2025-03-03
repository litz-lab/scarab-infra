#!/bin/bash
APPNAME="$1"

if [ "$APPNAME" == "mysql" ]; then
  cd $tmpdir/sysbench && ./autogen.sh && ./configure && make -j && sudo make install
  export MYSQLPW=$(sudo sed -n -e 's/^.*password = //p' /etc/mysql/debian.cnf | head -1)
  sudo service mysql start
  sudo -u mysql mysql --user=debian-sys-maint --password=$MYSQLPW --execute="CREATE DATABASE sbtest;CREATE USER sbtest;GRANT ALL PRIVILEGES ON sbtest.* TO sbtest;"
  cd $tmpdir/sysbench && sysbench ./src/lua/oltp_point_select.lua --mysql-host=127.0.0.1 --mysql-port=3306 --mysql-user=sbtest --db-driver=mysql --tables=10 --table-size=10000 prepare
  sudo service mysql stop
elif [ "$APPNAME" == "postgres" ]; then
  cd $tmpdir/sysbench && ./autogen.sh && ./configure --with-pgsql && make -j && sudo make install
  sudo service postgresql start
  sudo -u postgres psql -c "CREATE USER sbtest WITH PASSWORD 'password';" -c "CREATE DATABASE sbtest;" -c "GRANT ALL PRIVILEGES ON DATABASE sbtest TO sbtest;"
  cd $tmpdir/sysbench && sysbench ./src/lua/oltp_point_select.lua --pgsql-host=127.0.0.1 --pgsql-port=5432 --mysql-user=sbtest --pgsql-password=password --db-driver=pgsql --tables=10 --table-size=10000 prepare
  sudo service postgresql stop
fi
