// Nueva versión de las cajas de código
#let frame(title: none, metadata: (:), line_count: 0, body) = {
  let stroke = 0.5pt + black // Línea fina estilo KACTL
  
  block(width: 100%, breakable: true)[
    // Cabecera: Nombre del archivo y número de líneas
    #stack(
      dir: ltr,
      spacing: 1fr,
      text(weight: "bold", size: 1.1em, title),
      text(size: 0.8em, fill: gray)[#line_count lines]
    )
    
    // Bloque de Metadatos (Description, Time, etc.)
    #set text(size: 0.85em)
    #v(2pt)
    #for (key, value) in metadata {
      [*#key:* #value ]
    }
    
    // Separador y Código
    #v(-4pt)
    #line(length: 100%, stroke: stroke)
    #v(-6pt)
    #block(width: 100%, inset: (left: 2pt), body)
    #v(4pt)
  ]
}

#let codeblock(file_path, lang) = {
  let content = read(file_path)
  
  let header_pattern = if lang == "cpp" { 
    regex("(?s)/\*+\s*(.*?)\s*\*+/") 
  } else { 
    regex("(?s)\"\"\"\s*(.*?)\s*\"\"\"") 
  }
  
  let header_match = content.match(header_pattern)
  let metadata = (:)
  let clean_code = content
  
  if header_match != none {
    let header_text = header_match.captures.at(0)
    let fields = ("Author", "Description", "Time", "Status", "Usage")
    
    // Procesamos línea a línea el interior del comentario
    let header_lines = header_text.split("\n")
    let current_field = ""
    
    for line in header_lines {
      // Limpiamos asteriscos iniciales de C++ y espacios
      let clean_line = line.replace(regex("^\s*\*+\s?"), "").trim()
      if clean_line == "" { continue }
      
      let found_new_field = false
      for field in fields {
        if clean_line.starts-with(field + ":") {
          current_field = field
          let val = clean_line.slice((field + ":").len()).trim()
          metadata.insert(current_field, val)
          found_new_field = true
          break
        }
      }
      
      // Lógica especial: Solo Description permite multilínea
      if not found_new_field and current_field == "Description" {
        let prev_val = metadata.at("Description", default: "")
        metadata.insert("Description", prev_val + " " + clean_line)
      }
    }
    
    clean_code = content.replace(header_pattern, "").trim()
  }

  let lines = clean_code.split("\n").len()

  frame(
    title: file_path.split("/").at(-1), 
    metadata: metadata,
    line_count: lines
  )[
    #set text(size: 10pt) // Tamaño mínimo requerido
    #raw(clean_code, lang: lang)
  ]
}

// #let codeblock(file_path, lang) = {
//   let content = read(file_path)
  
//   let header_pattern = if lang == "cpp" { 
//     regex("(?s)/\*+\s*(.*?)\s*\*+/") 
//   } else { 
//     regex("(?s)\"\"\"\s*(.*?)\s*\"\"\"") 
//   }
  
//   let header_match = content.match(header_pattern)
//   let metadata = (:)
//   let clean_code = content
  
//   if header_match != none {
//     // Extraemos el texto dentro del comentario
//     let header_text = header_match.captures.at(0)
    
//     // Lista de campos que queremos extraer
//     let fields = ("Author", "Description", "Time", "Status", "Usage") // "date"
//     for field in fields {
//       // Buscamos "Campo: Valor" dentro del bloque de comentario
//       let field_pattern = regex("(?m)^\s*\*?\s*" + field + ":\s*(.*)")
//       let m = header_text.match(field_pattern)
//       if m != none {
//         metadata.insert(field, m.captures.at(0).trim())
//       }
//     }
//     // Quitamos el comentario del código para el PDF
//     clean_code = content.replace(header_pattern, "").trim()
//   }

//   let lines = clean_code.split("\n").len()

//   frame(
//     title: file_path.split("/").at(-1), 
//     metadata: metadata,
//     line_count: lines
//   )[
//     #set text(size: 10pt)
//     #raw(clean_code, lang: lang)
//   ]
// }


// NOTA: Si no te gusta el diseño de los códigos y te gustaba el diseño
// anterior, descomenta el siguiente código, y comenta todo lo anterior :D
// #let frame(title: none, body) = {
//   let stroke = black + 1pt
//   let radius = 4pt
//   block(stroke: stroke, radius: radius)[
//     #if title != none {
//         block(
//           stroke: stroke,
//           inset: 0.5em,
//           below: 0em,
//           radius: (top-left: radius, bottom-right: radius),
//           title.split("/").at(-1), // ../cpp/template.cpp -> template.cpp
//         )
//     }
//     #block(
//       width: 100%,
//       inset: (rest: 0.5em),
//       body,
//     )
//   ]
// }

// #let codeblock(file_path, lang) = {
//   frame(title: file_path)[
//     #raw(read(file_path), lang: lang)
//   ]
// }

