FROM jingpengw/caffe

RUN apt install -y -qq python-tk
WORKDIR /opt
RUN git clone --depth 1 https://github.com/jingpengw/kaffe.git
WORKDIR kaffe
RUN pip install -r requirements.txt
WORKDIR /opt/kaffe/DataProvider/python 
RUN make

RUN git clone --depth 1 https://github.com/torms3/DataProvider.git

ENV PYTHONPATH $PYTHONPATH:/opt/kaffe/layers
ENV PYTHONPATH $PYTHONPATH:/opt/kaffe
