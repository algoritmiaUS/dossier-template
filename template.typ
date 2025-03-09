#let dossier(university: "",
             team_name: "",
             members: (), 
             date:datetime.today(),
             num_cols:3,
             body) = {
               
  // Set the document's basic properties.
  set document(title: "Dossier Python", author: members)
  
  set page(
    margin: 1cm,
    numbering: "1",
    number-align: start,
    flipped: true
  )
  
  set text(font: "Arial", lang: "es", size: 10pt)
  
  // Title row.
  align(center)[
    #v(5.5cm)
    #image("logo-US.png", width: 100pt)
    #block(text(2em, university))
    #block(text(2.5em, team_name))
    #block(text(2em, members.join(", ")))
    #v(5.5cm)
    #block(text(2em, date.display()))
  ]
  pagebreak()

  // Main body.
  set par(justify: true)
  show: columns.with(num_cols, gutter: 1.3em)
  set heading(numbering: "1.")
  outline(depth: 1)
  body
}