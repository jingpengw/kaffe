FROM jingpengw/caffe

WORKDIR /opt
RUN git clone https://github.com/jingpengw/kaffe.git
WORKDIR kaffe

RUN git clone https://github.com/torms3/DataProvider.git

ENV PYTHONPATH $PYTHONPATH:/opt/kaffe/layers
ENV PYTHONPATH $PYTHONPATH:/opt/kaffe
