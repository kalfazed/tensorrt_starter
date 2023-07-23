# For new user: here are tips you need to do !!
## 1. modify your environment variables, such as $PATH, $LD_LIBRARY_PATH, $PKG_CONFIG_PATH, $GOPATH, and so on
## 2. set your own alias. I have demonstrated some useful alias I use everyday. Please change them as you wish :)
## 3. set up your conda env. After you install miniconda or anaconda, you should run "conda init fish" firstly.
## 4. Install fisher and z following:
##    https://github.com/jorgebucaran/fisher
##    https://github.com/jethrokuan/z
## 5. You can also change your prompt and keybinding in ./functions
# Have fun!!

if status is-interactive
    # Commands to run in interactive sessions can go here
end

export LSCOLORS=gxfxcxdxbxegedabagacad

set fish_greeting "welcome back!!"

# color setting
set -gx TERM xterm-256color

# theme
set -g theme_hide_hostname no
set -g theme_color_scheme terminal-dark
set -g fish_prompt_pwd_dir_length 1
set -g theme_display_user yes
set -g theme_hostname always

# aliases
alias ls     "ls -p -G"
alias la     "ls -A"
alias sleep  "sudo systemctl suspend"
alias ncu    "sudo /usr/local/cuda/bin/ncu"
alias make   "bear -- make -j64"
alias nsys   "sudo /usr/local/cuda/bin/nsys"
alias obs    "flatpak run com.obsproject.Studio"
alias gt     "git log --graph --pretty=format:'%x09%C(auto) %h %Cgreen %ar %Creset%x09 %C(cyan ul)%an%Creset %x09%C(auto)%s %d'"

# additional alias for exa
if type -q exa
  alias ll "exa --icons --long --no-user"
  alias lla "ll -a"
end

# set customized neovim as default editor
command -qv lvim && alias vim lvim
set -gx EDITOR lvim

# combine cd and ls together
functions --copy cd standard_cd
function cd
  standard_cd $argv; and ll
end

# Please modify your path here
set -gx PATH bin $PATH
set -gx PATH ~/.local/bin $PATH
set -gx PATH /usr/local/bin $PATH
set -gx PATH /usr/local/cuda/bin $PATH
set -gx PATH /home/kalfazed/packages/TensorRT-8.5.1.7/bin $PATH
set -gx LD_LIBRARY_PATH /home/kalfazed/packages/TensorRT-8.5.1.7/lib $LD_LIBRARY_PATH
set -gx LD_LIBRARY_PATH /usr/local/cuda/lib64 $LD_LIBRARY_PATH
set -gx LD_LIBRARY_PATH /usr/local/lib $LD_LIBRARY_PATH
set -gx PKG_CONFIG_PATH /usr/local/lib/pkgconfig $PKG_CONFIG_PATH

# NodeJS
set -gx PATH node_modules/.bin $PATH

# Go
set -g GOPATH $HOME/go
set -gx PATH $GOPATH/bin $PATH

# Please modify your OpenAI API key here to enable ChatGPT, replace xxx to your Key
set -gx OPENAI_API_KEY xxx $OPENAI_API_KEY

# cuda lazy loading
set -g CUDA_MODULE_LOADING LAZY 

set LOCAL_CONFIG (dirname (status --current-filename))/config-local.fish
if test -f $LOCAL_CONFIG
  source $LOCAL_CONFIG
end

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
if test -f /home/kalfazed/miniconda3/bin/conda
    eval /home/kalfazed/miniconda3/bin/conda "shell.fish" "hook" $argv | source
end
# <<< conda initialize <<<

# Please create your own conda env and actvate it here
conda activate trt-starter
