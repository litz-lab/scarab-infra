# syntax = edrevo/dockerfile-plus
FROM ubuntu:20.04
RUN apt-get update && apt-get install -y --no-install-recommends nano && rm -rf /var/lib/apt/lists/*;
RUN apt-get update -y && apt-get install -y --no-install-recommends openjdk-17-jdk-headless \
        && rm -rf /var/lib/apt/lists/*

INCLUDE+ ../common/Dockerfile.common
USER root

RUN apt-get update -y \
	&& apt-get install -y --no-install-recommends procps telnet lsof wget unzip \
	&& rm -rf /var/lib/apt/lists/*

COPY ./workloads/solr/files/limits.txt /root/.
RUN cat /root/limits.txt >> /etc/security/limits.conf

ENV BASE_PATH /usr/src
ENV SOLR_VERSION 9.1.1
ENV SOLR_HOME $BASE_PATH/solr-$SOLR_VERSION
ENV PACKAGES_URL http://datasets.epfl.ch/cloudsuite
ENV INDEX_URL $PACKAGES_URL/index
ENV SCHEMA_URL $PACKAGES_URL/schema.xml
ENV SOLR_CONFIG_URL $PACKAGES_URL/solrconfig.xml
ENV SOLR_PORT 8983
ENV SOLR_CORE_DIR $BASE_PATH/solr_cores
ENV SERVER_HEAP_SIZE 3g
ENV NUM_SERVERS 1
ENV SERVER_0_IP localhost
ENV ZOOKEEPER_PORT $SOLR_PORT

#INSTALL SOLR
RUN cd $BASE_PATH \ 
  && wget --progress=bar:force -O solr.tar.gz "https://archive.apache.org/dist/solr/solr/$SOLR_VERSION/solr-$SOLR_VERSION.tgz" \ 
  && tar -zxf solr.tar.gz \
  && rm solr.tar.gz


RUN cd $SOLR_HOME/server/solr/configsets/ \
  && cp -r _default cloudsuite_web_search \
  && rm cloudsuite_web_search/conf/managed* 

COPY ./workloads/solr/files/schema.xml $SOLR_HOME/server/solr/configsets/cloudsuite_web_search/conf/.
COPY ./workloads/solr/files/solrconfig.xml $SOLR_HOME/server/solr/configsets/cloudsuite_web_search/conf/.

#RELOAD CONFIGURATION
RUN     cd $SOLR_HOME \
	&& mkdir -p $SOLR_CORE_DIR \
	&& cp -R server/solr/* $SOLR_CORE_DIR 

#INSTALL DOCKER
RUN apt-get update && apt-get install -y curl && curl -sSL https://get.docker.com/ | sh

COPY ./workloads/solr/workload_user_entrypoint.sh /usr/local/bin/workload_user_entrypoint.sh
RUN chmod +x /usr/local/bin/workload_user_entrypoint.sh

EXPOSE $SOLR_PORT

RUN sed -i "/SOLR_JETTY_HOST/c\SOLR_JETTY_HOST=\"0.0.0.0\"" ${SOLR_HOME}/bin/solr.in.sh 


RUN mkdir $SOLR_HOME/server/logs
RUN touch $SOLR_HOME/server/logs/solr_gc.log
# ENTRYPOINT "/entrypoint.sh" && /bin/bash
# RUN echo "/entrypoint.sh" >> ~/.bashrc
# RUN echo '(docker run -it --name web_search_client --net host cloudsuite/web-search:client $(hostname -I) 10 && pkill java) &' >> ~/.bashrc
CMD ["/bin/bash"]
