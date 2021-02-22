FROM public.ecr.aws/lambda/python:latest

RUN yum update

RUN pip3 install 'sagemaker>=2,<3'
RUN pip3 install numpy
RUN pip3 install boto3
RUN pip3 install Pillow
RUN pip3 install matplotlib
RUN pip3 install pandas

COPY code/ss_inference.py /var/task

CMD [ "ss_inference.handler" ]


