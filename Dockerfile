FROM jingpengw/caffe
RUN apt-get update
RUN apt-get install -y -qq python-tk
WORKDIR /opt
RUN git clone --depth 1 https://github.com/jingpengw/kaffe.git
WORKDIR kaffe
RUN pip install -r requirements.txt

RUN git clone --depth 1 https://github.com/torms3/DataProvider.git
WORKDIR /opt/kaffe/DataProvider/python 
RUN make


ENV PYTHONPATH $PYTHONPATH:/opt/kaffe/layers
ENV PYTHONPATH $PYTHONPATH:/opt/kaffe
