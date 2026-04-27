# cmake-lint: disable=C0103,C0301,C0111,E1120,R0913,R0915

if(NOT WIN32)
  string(ASCII 27 Esc)
  set(ColorReset "${Esc}[m")
  set(ColourBold "${Esc}[1m")
  set(Underline "${Esc}[4m")
  set(Red "${Esc}[31m")
  set(Green "${Esc}[32m")
  set(Yellow "${Esc}[33m")
  set(Blue "${Esc}[34m")
  set(Magenta "${Esc}[35m")
  set(Cyan "${Esc}[36m")
  set(White "${Esc}[37m")
  set(BoldRed "${Esc}[1;31m")
  set(BoldGreen "${Esc}[1;32m")
  set(BoldYellow "${Esc}[1;33m")
  set(BoldBlue "${Esc}[1;34m")
  set(BoldMagenta "${Esc}[1;35m")
  set(BoldCyan "${Esc}[1;36m")
  set(BoldWhite "${Esc}[1;37m")
  set(DarkGray "${Esc}[1;90m")
  set(Dim "${Esc}[2m")
  set(StrikeBegin "${Esc}[9m")
  set(StrikeEnd "${Esc}[0m")
endif()

set(DOTTED_LINE_SYMBOL "${ColorReset}. . . . . . . . . . . . . . . .")
string(APPEND DOTTED_LINE_SYMBOL " . . . . . . . . . . . . . . . . . . . . ")

set(DASHED_LINE_SYMBOL "${ColorReset}.................................")
string(APPEND DASHED_LINE_SYMBOL "...................................... ")

set(ON_OFF_VALUES "ON" "OFF")

function(PureLength Text Result)
  set(rt ${Text})
  string(FIND ${rt} "${Magenta}" mg_fnd)

  if(mg_fnd GREATER -1)
    string(REGEX REPLACE "${Esc}\\[35m" "" rt ${rt})
  endif()

  string(LENGTH "${rt}" TextLength)
  set(${Result}
      "${TextLength}"
      PARENT_SCOPE)
endfunction()

function(PadTo Text Padding Target Result)
  purelength("${Text}" TextLength)
  math(EXPR PaddingNeeded "${Target} - ${TextLength}")
  set(rt ${Text})

  if(PaddingNeeded GREATER 0)
    foreach(i RANGE 0 ${PaddingNeeded})
      set(rt "${rt}${Padding}")
    endforeach()
  else()
    set(rt "${rt}")
  endif()

  set(${Result}
      "${rt}"
      PARENT_SCOPE)
endfunction()

function(
  PrintChoices
  Label
  Flag
  Choices
  Value
  Default
  Color
  OutputString
  Padding)
  set(rstring "- ${Label}")

  if(NOT "${Flag}" STREQUAL "")
    string(APPEND rstring " [${Magenta}${Flag}${ColorReset}]")
  endif()

  string(APPEND rstring ":")

  if(${Padding} EQUAL 0)
    set(longest 0)
    foreach(ch IN LISTS Choices)
      string(LENGTH "${ch}" clen)
      if(clen GREATER longest)
        set(longest ${clen})
      endif()
    endforeach()

    if(longest GREATER 20)
      set(ncols 3)
    else()
      set(ncols 4)
    endif()
    math(EXPR lastcol "${ncols} - 1")

    set(col_pos 0)
    foreach(ch IN LISTS Choices)
      if(col_pos EQUAL 0)
        string(APPEND rstring "\n    ")
      endif()

      set(col "${Dim}")
      if(NOT "${Value}" STREQUAL "")
        list(FIND Value "${ch}" _idx)
        if(_idx GREATER -1)
          set(col "${Color}")
        endif()
      endif()
      if(NOT "${Default}" STREQUAL "" AND "${ch}" STREQUAL "${Default}")
        set(col "${Underline}${col}")
      endif()

      string(APPEND rstring "${col}~ ${ch}${ColorReset}")

      if(NOT col_pos EQUAL lastcol)
        string(LENGTH "${ch}" clen)
        math(EXPR pad "${longest} - ${clen} + 4")
        foreach(i RANGE 0 ${pad})
          string(APPEND rstring " ")
        endforeach()
      endif()

      math(EXPR col_pos "(${col_pos} + 1) % ${ncols}")
    endforeach()
  else()
    padto("${rstring}" " " ${Padding} rstring)

    set(parts "")
    foreach(ch IN LISTS Choices)
      if("${ch}" STREQUAL "ON")
        set(ch_col "${Green}")
      elseif("${ch}" STREQUAL "OFF")
        set(ch_col "${Red}")
      else()
        set(ch_col "${Color}")
      endif()

      set(elem "${Dim}${ch}${ColorReset}")
      if(NOT "${Value}" STREQUAL "")
        list(FIND Value "${ch}" _idx)
        if(_idx GREATER -1)
          set(elem "${ch_col}${ch}${ColorReset}")
        endif()
      endif()
      if(NOT "${Default}" STREQUAL "" AND "${ch}" STREQUAL "${Default}")
        set(elem "${Underline}${elem}${ColorReset}")
      endif()

      list(APPEND parts "${elem}")
    endforeach()

    string(JOIN "/" joined ${parts})
    string(APPEND rstring "${joined}")
  endif()

  set(${OutputString}
      "${rstring}"
      PARENT_SCOPE)
endfunction()
