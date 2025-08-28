#import "template.typ": dossier 

#show: dossier.with(
  university: "Universidad de Sevilla",
  team_name: "Teorema del Sándwich de Ham",
  members: (
    "Kenny Flores",
    "Pablo Dávila",
    "Pablo Reina"
  ),
  date: datetime.today(),
  num_cols:2,
  flipped:false // Indica si la orientación es vertical u horizontal
)

// https://forum.typst.app/t/can-i-configure-my-document-e-g-draft-release-version-color-theme-when-creating-a-pdf-without-modifying-the-typst-file-directly/160
#let language = {
  let valid-values = ("python", "cpp", "java")
  let value = sys.inputs.at("language", default: "python") // CAMBIAR LENGUAJE AQUÍ
                                                           // PARA VISUALIZACIÓN
  assert(value in valid-values, message: "`--input language` must be in {valid-values}")
  value
}

#include language+".typ"
