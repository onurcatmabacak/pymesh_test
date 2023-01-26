FROM pymesh/pymesh

WORKDIR /home

RUN apt-get install openssh-client -y
COPY id_rsa.pub /home/.ssh/id_rsa.pub
RUN echo "git clone git@github.com:onurcatmabacak/pymesh_test.git" > /home/get_pymesh_test.sh