FROM ubuntu:20.04
ARG DEBIAN_FRONTEND=noninteractive
ARG USERNAME
ENV DISPLAY=host.docker.internal:0.0

RUN apt-get update && apt-get install -y \
    sudo bc zsh git vim wget curl \
    libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev \
    libglu1-mesa-dev freeglut3-dev mesa-common-dev \
    clang-tools cmake doxygen python3 python3-pip build-essential

RUN ln -s /bin/python3 /bin/python
RUN pip install autopep8

RUN apt-get clean && \
  apt-get autoclean && \
  apt-get autoremove -y && \
  rm -rf /var/lib/cache/* && \
  rm -rf /var/lib/log/*

RUN useradd -ms /bin/zsh $USERNAME
RUN usermod -aG sudo $USERNAME
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER $USERNAME
WORKDIR /home/$USERNAME
RUN chown $USERNAME:$USERNAME /home/$USERNAME

RUN git clone https://github.com/tarjoilija/zgen.git "${HOME}/.zgen"
RUN echo 'source "${HOME}/.zgen/zgen.zsh"\n\
zgen load denysdovhan/spaceship-zsh-theme spaceship\n\
alias ls="ls --color=auto"' >> "${HOME}/.zshrc"

ENTRYPOINT ["tail", "-f", "/dev/null"]
