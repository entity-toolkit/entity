rtouch() {
  mkdir -p $(sed 's/\(.*\)\/.*/\1/' <<<$1) && touch $1
}

function runcommand {
  if [ $deploy = "OFF" ]; then
    echo ": $1"
  else
    if [ $verbose = "ON" ]; then
      eval "$1"
    else
      eval "$1" >>${logfile} 2>>${logfile}
    fi
  fi
}

declare -x FRAME
declare -x FRAME_INTERVAL

set_spinner() {
  case $1 in
  spinner1)
    FRAME=("⠋" "⠙" "⠹" "⠸" "⠼" "⠴" "⠦" "⠧" "⠇" "⠏")
    FRAME_INTERVAL=0.1
    ;;
  spinner2)
    FRAME=("-" "\\" "|" "/")
    FRAME_INTERVAL=0.25
    ;;
  spinner3)
    FRAME=("◐" "◓" "◑" "◒")
    FRAME_INTERVAL=0.5
    ;;
  spinner4)
    FRAME=(":(" ":|" ":)" ":D")
    FRAME_INTERVAL=0.5
    ;;
  spinner5)
    FRAME=("◇" "◈" "◆")
    FRAME_INTERVAL=0.5
    ;;
  spinner6)
    FRAME=("⚬" "⚭" "⚮" "⚯")
    FRAME_INTERVAL=0.25
    ;;
  spinner7)
    FRAME=("░" "▒" "▓" "█" "▓" "▒")
    FRAME_INTERVAL=0.25
    ;;
  spinner8)
    FRAME=("☉" "◎" "◉" "●" "◉")
    FRAME_INTERVAL=0.1
    ;;
  spinner9)
    FRAME=("❤" "♥" "♡")
    FRAME_INTERVAL=0.15
    ;;
  spinner10)
    FRAME=("✧" "☆" "★" "✪" "◌" "✲")
    FRAME_INTERVAL=0.1
    ;;
  spinner11)
    FRAME=("●" "◕" "☯" "◔" "◕")
    FRAME_INTERVAL=0.25
    ;;
  *)
    echo "No spinner is defined for $1"
    exit 1
    ;;
  esac
}

function is_gpu_arch {
  local ar=$1
  if [[ $ar == "VOLTA"* ]] || [[ $ar == "TURING"* ]] || [[ $ar == "AMPERE"* ]] || [[ $ar == "MAXWELL"* ]] || [[ $ar == "PASCAL"* ]] || [[ $ar == "KEPLER"* ]] || [[ $ar == "INTEL"* ]] || [[ $ar == "VEGA"* ]] || [[ $ar == "NAVI"* ]]; then
    echo "TRUE"
  else
    echo "FALSE"
  fi
}

GRAY='\033[0;30m'
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'
