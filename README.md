# Subtren Estudio

Editor de música que corre en el navegador. No hay que instalar nada.

La separación de voz con inteligencia artificial necesita que la página se sirva desde una web (por ejemplo GitHub Pages); si abres `index.html` directo del disco, todo lo demás funciona pero la IA no.

## Publicarlo en GitHub Pages

1. En GitHub, ve a **Settings → General** y, al final, en **Danger Zone**, cambia el repo a **Public** (GitHub Pages gratis solo funciona con repos públicos).
2. Ve a **Settings → Pages**. En **Source** escoge **Deploy from a branch**, la rama `editor-de-musica` (o `main` si ya está unida) y la carpeta `/ (root)`. Guarda.
3. En uno o dos minutos la página queda en `https://ant939c-web.github.io/subtren/`.

## Qué puedes hacer

- **Abrir canciones** (MP3, WAV, M4A, OGG…) o arrastrarlas a la página. Trae un beat de dembow de ejemplo para probar.
- **Editar**: cortar, copiar, pegar, pegar encima (mezclar), borrar, dejar solo un pedazo, silenciar, deshacer y rehacer.
- **Unir canciones**: añadir otra canción al final.
- **Volumen y forma**: ganancia, normalizar, fade in, fade out, al revés y cambio de velocidad.
- **Efectos**: eco, reverb, slowed + reverb, más bajo, más brillo, bajo el agua, teléfono, distorsión, robot, audio 8D, ardilla y demonio.
- **Loop** de la selección, zoom en la onda y atajos de teclado.
- **Masterizar** la canción con un botón: estilos Suave, Con fuerza y Radio, volumen final en LUFS y comparación con el original.
- **Voz y pista**: con inteligencia artificial (Demucs) deja solo la voz en el pedazo que escojas, con la pista parada, o quita la voz para karaoke. Corre en el navegador, sin servidor. La primera vez baja 174 MB y después queda guardada.
- **Exportar** el resultado como WAV.

Los efectos se aplican a lo que tengas seleccionado; si no hay nada seleccionado, a toda la canción.

## Créditos y licencias

- Separación de voz: [Demucs](https://github.com/facebookresearch/demucs) de Meta (MIT), versión en JavaScript de [demucs-js](https://github.com/bakkot/demucs-js) (MIT, carpeta `demucs/`).
- Los pesos del modelo (`modelo/`, htdemucs) son de Meta y solo se permiten para **uso personal y de investigación**.
- Motor: [ONNX Runtime Web](https://github.com/microsoft/onnxruntime) 1.23.0 de Microsoft (MIT, carpeta `ort/`).
