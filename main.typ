#import "template.typ": dossier 
// Take a look at the file `template.typ` in the file panel
// to customize this template and discover how it works.

#show: dossier.with(
  university: "Universidad de Sevilla",
  team_name: "Teorema del Sándwich de Ham",
  members: (
    "Kenny Flores",
    "Pablo Dávila",
    "Pablo Reina"
  ),
  date: datetime.today(),
  num_cols:3
)

#let language = "python" //Puedes cambiar por 

// Lógica condicional para mostrar contenido del módulo
#if language == "python" {
  include "python.typ"
} else if language == "kotlin" {
  include "kotlin.typ"
} else if language == "cpp" {
  include "cpp.typ"
} else {
  [Error: Tienes que poner un nombre válido.]
}