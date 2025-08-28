// Modificado de: https://github.com/typst/typst/issues/1494
#let frame(title: none, body) = {
  let stroke = black + 1pt
  let radius = 4pt
  block(stroke: stroke, radius: radius)[
    #if title != none {
        block(
          stroke: stroke,
          inset: 0.5em,
          below: 0em,
          radius: (top-left: radius, bottom-right: radius),
          title.split("/").at(-1), // ../cpp/template.cpp -> template.cpp
        )
    }
    #block(
      width: 100%,
      inset: (rest: 0.5em),
      body,
    )
  ]
}

#let codeblock(file_path, lang) = {
  frame(title: file_path)[
    #raw(read(file_path), lang: lang)
  ]
}

