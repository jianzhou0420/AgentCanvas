[English](../README.md) | [中文](README_zh.md) | **Español** | [日本語](README_ja.md) | [한국어](README_ko.md)

<div align="center">

# AgentCanvas

### Automating the Design of Embodied Agent Architectures

**Jian Zhou · Sihao Lin · Jin Li · Shuai Fu · Gengze Zhou · Qi Wu**

Australian Institute for Machine Learning, University of Adelaide

<p>
  <a href="https://arxiv.org/abs/2606.30111"><img src="https://img.shields.io/badge/arXiv-2606.30111-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv"></a>
  <a href="https://jianzhou0420.github.io/src/works/AgentCanvas/index.html"><img src="https://img.shields.io/badge/Project%20Page-1f6feb?style=for-the-badge&logo=googlechrome&logoColor=white" alt="Project Page"></a>
  <a href="https://jianzhou0420.github.io/src/works/AgentCanvas/paper.html"><img src="https://img.shields.io/badge/Paper%20Page-1f6feb?style=for-the-badge&logo=googlechrome&logoColor=white" alt="Paper Page"></a>
  <a href="https://jianzhou0420.github.io/AgentCanvas/"><img src="https://img.shields.io/badge/Docs-2ea44f?style=for-the-badge&logo=readthedocs&logoColor=white" alt="Documentation"></a>
  <a href="#9-citación"><img src="https://img.shields.io/badge/BibTeX-Cite-4285F4?style=for-the-badge&logo=googlescholar&logoColor=white" alt="BibTeX"></a>
</p>

<img src="../assets/readme/editor-hero.gif" alt="Editor de AgentCanvas: el ejecutor MapGPT se carga como un grafo de nodos y cables, luego un episodio R2R en vivo se ejecuta de extremo a extremo" width="760">

<sub><em>Grabado en vivo en el editor — el ejecutor MapGPT se carga, luego un episodio real de R2R se ejecuta de extremo a extremo.</em></sub>

</div>

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](../LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-green.svg)](https://www.python.org/)
[![Status: Research Preview](https://img.shields.io/badge/Status-Research_Preview-orange.svg)](#7-estado-del-proyecto)
[![GitHub stars](https://img.shields.io/github/stars/jianzhou0420/AgentCanvas?style=social)](https://github.com/jianzhou0420/AgentCanvas/stargazers)

**Una plataforma visual de diseño de agentes para la investigación en IA encarnada.** Un grafo tipado, dos roles: un *banco de pruebas* que ejecuta agentes encarnados, y un *andamiaje* que los agentes de programación editan y verifican.

AgentCanvas permite a los investigadores prototipar agentes encarnados — para VLN, EQA, VLA y tareas adyacentes — dibujando grafos de nodos que se ejecutan en tiempo real contra simuladores (Habitat-Sim, MatterSim, SAPIEN/ManiSkill2, MuJoCo/robosuite) o, en principio, configuraciones del mundo real. *Un JSON = un agente = un grafo*: el comportamiento del agente es un grafo de flujo de datos, no código imperativo; el grafo es la fuente de verdad, guardado como un único archivo JSON y cargado como un agente completo.

**Diseñado para**: investigadores que quieren componer, comparar y compartir arquitecturas de agentes encarnados sin reescribir la pila de ejecución cada vez. La plataforma cubre VLN (Vision-and-Language Navigation), EQA (Embodied Question Answering), benchmarks de políticas VLA (Vision-Language-Action), y se adapta a otros entornos encarnados / agénticos mediante el modelo de nodeset.

> **Estado**: Vista previa de investigación, en desarrollo activo · 46 ADRs · más de 40 nodesets en cuatro paletas intercambiables — **env** (simuladores), **method** (bucles de razonamiento), **model** (modelos fundacionales), **policy** (controladores neuronales) · editor de lienzo, ejecutor de grafos con iteración multi-ámbito, contenedores de estado, nodesets en modo servidor auto-alojados, sistema de hooks, JobScheduler de subproceso-por-ejecución + grupo de workers + inferencia por lotes, y un bus de errores unificado — todo en producción.

> **Versionado**: pre-1.0 (v0.x). v1.0 se publicará cuando la API pública sea estable (de código abierto + congelada bajo SemVer) — independiente de cualquier artículo. Consulta la [Política de Versionado](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/repo/versioning.html).

> **Contribuir**: Dos tipos, ambos bienvenidos. **Contenido** — escribe un nodeset (herramienta o método) o compón un grafo, mediante PR a `workspace/`; se te acredita en el tablero de [Créditos](#créditos), con un enlace de cita si tiene un artículo. **Core** — mejora el framework (UI, backend, funcionalidades, refactorizaciones); abre primero una [Discussion](https://github.com/jianzhou0420/AgentCanvas/discussions) para cualquier cosa grande. Consulta [CONTRIBUTING.md](../CONTRIBUTING.md).

---

## Contenidos

1. [¿Por qué AgentCanvas?](#1-por-qué-agentcanvas) — un sustrato buscable para agentes encarnados, y los puntos débiles que resuelve
2. [Características](#2-características) — los principios *un JSON = un agente* (§2.2) / *una clase de Python = un nodo* (§2.6), además del editor de lienzo, ejecutor de grafos, entornos de ejecución aislados, grafos anidados, contenedores de estado, hooks
3. [Ruta de Sim a Real](#3-ruta-de-sim-a-real) — el mismo grafo de agente en el simulador hoy, robot real mañana — mediante env-como-nodeset + modo servidor + ROS
4. [Primeros Pasos](#4-primeros-pasos) — requisitos previos, ejecutar el panel web, ejecutar una evaluación, ejecutar la búsqueda de arquitectura, servir la documentación
5. [Arquitectura](#5-arquitectura) — frontend · backend · workspace · simuladores
6. [Estructura del Proyecto](#6-estructura-del-proyecto) — mapa de directorios de nivel superior
7. [Estado del Proyecto](#7-estado-del-proyecto) — versiones: v0.1 experimentos → v0.2 vista previa → v1.0 → v2.0
8. [Contribuir](#8-contribuir) — dónde se necesita más ayuda · créditos
9. [Citación](#9-citación) — cómo citar AgentCanvas
10. [Licencia](#10-licencia) — Apache 2.0

---

## 1. ¿Por qué AgentCanvas?

Los agentes encarnados — que abarcan VLN, EQA y VLA — se construyen cada vez más componiendo modelos fundacionales con percepción, mapeo, memoria, planificación y acción. A diferencia de las políticas de extremo a extremo, cuya estructura queda absorbida en los pesos, esta arquitectura es *explícita y editable*. Eso plantea la pregunta en torno a la cual se construye AgentCanvas — **¿puede el diseño de agentes buscarse en lugar de construirse a mano?** — junto con dos pilas de problemas que debe superar en el camino.

<details>
<summary><b>La arquitectura de agentes se construye a mano — y podría buscarse</b></summary>

<br>

Cada agente fija una elección en cada unión — abstracciones de sensores, representaciones de mapas, estado de memoria, estructura del prompt, topología del planificador, ubicación del modelo, interfaces de acción — a mano, normalmente para un único benchmark. A medida que los modelos fundacionales y las herramientas encarnadas se multiplican, el espacio crece más rápido de lo que la iteración manual puede cubrir, así que el movimiento natural es buscarlo en lugar de ajustarlo a mano.

La Búsqueda de Arquitectura de Agentes (AAS) ya hace esto para agentes del dominio del texto, pero la transferencia a lo encarnado no es gratuita: simuladores con estado, puntuación ruidosa multi-episodio, trazas largas de percepción/acción, y ninguna paleta lista para usar de primitivas encarnadas. AgentCanvas es nuestro intento de suministrar el sustrato que falta — un andamiaje que un agente de programación puede leer, editar, ejecutar y verificar — para que buscar el diseño de agentes también sea posible para los agentes encarnados.

</details>

<details>
<summary><b>Puntos débiles específicos de lo encarnado</b></summary>

<br>

- **La pila encarnada moderna es densa** — un agente encarnado funcional necesita razonamiento LLM + uso de herramientas + acoplamiento con el simulador + herramientas espaciales, todo cableado en conjunto. Construir esto desde cero por proyecto es prohibitivamente costoso, y la mayor parte del esfuerzo se dedica a la capa de ejecución en lugar de a la idea que se está probando.
- **Pesadilla de ingeniería** — un agente encarnado no es un modelo sino todo un sistema — un simulador con estado más una pila de modelos y herramientas pesados. Solo ejecutarlo, ya no digamos a la escala que requiere el benchmarking, es de por sí un trabajo de ingeniería arduo:
  - **Infierno de entornos Python** — ningún entorno Python único satisface cada parte; cada simulador, VLM, detector y política fija su propia versión incompatible de CUDA / torch / Python, así que encontrar un único runtime que todos compartan suele ser imposible — acabas manteniendo varios entornos incompatibles solo para cargar el agente.
  - **Batching** — el simulador de cada worker es un proceso con estado separado que avanza a su propio ritmo; puedes agrupar el modelo pero no los simuladores, así que cada paso se convierte en una danza asíncrona de recolectar-observaciones → inferir-en-lote → dispersar-acciones.
  - **Otra infraestructura** — trayectorias multimodales que deben registrarse y poder reproducirse, checkpoint/reanudación de ejecuciones de GPU de varias horas que *van a* fallar, y depuración a través de los límites entre procesos.

  A lo largo del ciclo de investigación de un solo artículo, el investigador paga demasiado de este coste de ingeniería en lugar de centrarse en el algoritmo en sí.
- **Dependencias ocultas de ground-truth** — muchos métodos dependen silenciosamente del ground truth proporcionado por el simulador (poses de objetos, etiquetas semánticas, navegabilidad) en lugar de la percepción real. A veces es una forma legítima de controlar el experimento — pero, sea descuido o no, a menudo no se menciona en el artículo.

</details>

<details>
<summary><b>Puntos débiles comunes de la investigación en IA (amplificados aquí)</b></summary>

<br>

- **Implementaciones no reproducibles** — cada artículo construye su agente desde cero con una base de código distinta; comparar métodos de forma justa o reproducir resultados es doloroso — y muchos de ellos son **`Code coming SOON`** (**S**omeday, **O**r **O**bviously **N**ever — "pronto", es decir: algún día, o evidentemente nunca).
- **Artículo ≠ código** — los artículos muestran diagramas de flujo limpios, pero el código real diverge de formas no documentadas. Reproducir un artículo significa hacer ingeniería inversa de su implementación.
- **Código fuertemente acoplado** — la lógica de dominio (prompts, herramientas, políticas) está enredada con la infraestructura. Cambiar un componente significa reescribir el pipeline.

</details>

---

## 2. Características

> **Referencia completa en la documentación** — la mayoría de las funcionalidades de abajo tienen una página de implementación (mecanismo · archivos clave · estado actual): **[Las Nueve Capacidades →](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/capabilities/index.html)**

### 2.1 Editor de Lienzo Visual

Un espacio de trabajo plano al estilo ComfyUI donde coexisten todos los tipos de nodos — entornos, LLMs, cadenas de razonamiento, compuertas de control y visores de salida. Arrastra nodos desde la barra lateral, conéctalos entre sí, pulsa Play.

### 2.2 Motor de Ejecución de Grafos

**Un JSON = un agente.** Todo el comportamiento de un agente — nodos, cableado, configuraciones, contenedores de estado, hooks — es un único archivo JSON: cárgalo, ejecútalo, compártelo, haz un diff. Sin código de pipeline oculto; lo que ves en el lienzo es lo que se ejecuta.

```jsonc
// Simplificado — los grafos reales incluyen contenedores de estado, hooks y más nodos
{
  "name": "NavGPT-CE",
  "description": "VLN reasoning graph with planner, VLM, and navigation memory",
  "kind": "graph",
  "nodes": [
    { "id": "observe", "type": "env_habitat__observe_egocentric", "config": {} },
    { "id": "planner", "type": "llmCall",                         "config": { "temperature": 0.0 } },
    { "id": "step",    "type": "env_habitat__step_discrete",      "config": {} }
  ],
  "edges": [
    { "source": "observe", "sourceHandle": "rgb", "target": "planner", "targetHandle": "image" },
    { "source": "planner", "sourceHandle": "action", "target": "step", "targetHandle": "action" }
  ]
}
```

El motor entonces ejecuta ese grafo: los nodos se disparan cuando llegan sus entradas, no en un orden fijo. El mismo motor maneja cada forma de grafo que AgentCanvas v1 admite — consulta [`major-versions.html`](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/core/major-versions.html) para la lista completa de formas de agente cubiertas por el paradigma de topología-estática-acotada de v1:

- **Flujos de trabajo DAG** — una sola pasada hacia adelante para pipelines acíclicos
- **Bucles de agente cíclicos** — observar-pensar-actuar-repetir mediante un modelo de **dos pivotes**: un **`IterIn`** de dos lados (entradas de inicialización al arranque de la ejecución a su izquierda, acarreo de bucle por iteración a su derecha) más **`IterOut`**, manteniendo el grafo visualmente acíclico a la vez que habilita ciclos en tiempo de ejecución (ADR-dataflow-008, que plegó el anterior modelo de tres pivotes `initialize`/IterIn/IterOut de ADR-dataflow-006 a dos)
- **Iteración multi-ámbito** — N pares `(IterIn, IterOut)` coexistiendo en un único grafo plano (ADR-dataflow-007 / ADR-executor-003)
- **Bucles ReAct** — ya sea ocultos dentro de una subclase `LLMCallNode` o expresados explícitamente como router + N ramas de herramientas predeclaradas
- **Multi-agente acotado** — fan-out de N-fijo o acotado por `K_max` (p. ej., debate al estilo DiscussNav, roles fijos al estilo AutoGen)
- **Plan-and-Execute** — sobre un pool de herramientas acotado, despachado por router

### 2.3 Entornos de Ejecución Aislados

Las herramientas de investigación a menudo necesitan entornos Python en conflicto (Habitat necesita Python 3.8, SLAM necesita ROS). Cualquier `BaseNodeSet` puede ejecutarse en **modo servidor** — el framework auto-genera un servidor HTTP a partir de las definiciones de puertos del nodeset, ejecutándose en su propio intérprete. Cero código adicional:

```
# Mismo código de nodeset, dos modos de despliegue:
POST /api/components/nodesets/env_habitat/load              # en proceso
POST /api/components/nodesets/env_habitat/load?mode=server  # proceso separado
```

### 2.4 Sistema de Grafos Anidados

Guarda cualquier grafo del lienzo como un **graph node** y arrástralo a otro lienzo como un bloque reutilizable. Esto habilita arquitecturas de agentes jerárquicas — un planificador de alto nivel que contiene graph nodes de subagentes. Semántica de instantánea: cada instancia es una copia profunda.

### 2.5 Sistema de Contenedores de Estado

Estado persistente compartido a través de las iteraciones del bucle del agente mediante una arquitectura de doble cableado:

- Las **aristas de datos** transportan el flujo de datos entre nodos (IMAGE, TEXT, ACTION, POSE, …)
- Las **concesiones de acceso** permiten a los nodos leer/escribir **StateContainers** — elementos visibles del lienzo con entradas nombradas, reductores configurables (Accumulator, LastWrite, Counter), y un eje de **Lifetime** (`forever` / `step` / `episode` / `run` / `custom`) que limpia automáticamente la memoria en el límite de señal correcto (ADR-dataflow-002, ADR-dataflow-004)

→ [Documento de diseño de State Containers](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/design-docs/graph/state-containers.html)

### 2.6 Nodos Definidos en Python

**Una clase de Python = un nodo.** Cada nodo del lienzo — herramientas, entornos, habilidades, políticas — es una única clase de Python: declara los puertos, implementa `forward()`, deja el archivo en `workspace/`, y la plataforma lo auto-descubre. Sin cambios en el framework, sin TypeScript, sin código repetitivo de registro.

```python
from app.components import BaseCanvasNode, PortDef

class MeasureDistanceNode(BaseCanvasNode):
    node_type    = "basic_agent__measure_distance"
    display_name = "Measure Distance"
    description  = "Euclidean distance between two 3D positions"
    category     = "tool"
    icon         = "Ruler"

    input_ports  = [
        PortDef("pos_a", "TEXT", "Position A as [x, y, z]"),
        PortDef("pos_b", "TEXT", "Position B as [x, y, z]"),
    ]
    output_ports = [
        PortDef("distance", "TEXT", "Euclidean distance (meters)"),
    ]

    async def forward(self, inputs, ctx):
        a, b = parse_vec3(inputs["pos_a"]), parse_vec3(inputs["pos_b"])
        dist = math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))
        return {"distance": f"{dist:.2f}"}
```

El nodo entonces aparece en la barra lateral del lienzo y se conecta con cualquier otro nodo con tipos de puerto compatibles. Su apariencia también está gobernada por Python: `GenericBlockRenderer` renderiza cualquier nodo automáticamente a partir de `NodeUIConfig` — colores, disposición, controles de configuración en línea (sliders, desplegables, campos de texto) y widgets de visualización — así que no se necesita ningún componente React personalizado.

### 2.7 Sistema de Hooks

Comandos de shell se disparan antes/después de cada ejecución de nodo y en los límites del ciclo de vida del grafo. Los hooks pueden registrar salidas, validar entradas, bloquear nodos o modificar datos — todo sin cambiar los nodos del grafo. Los hooks viajan con los grafos guardados.

### 2.8 Evaluación por Lotes y Cola de Trabajos

El mismo grafo que se ejecuta en el lienzo puede enviarse como un trabajo de evaluación que lo puntúa sobre cientos de episodios. Un `JobScheduler` propiedad del backend controla la admisión frente a un presupuesto de VRAM compartido entre todas las sesiones (ADR-eval-003); cada ejecución admitida es su propio subproceso, así que los reinicios del backend no matan las evaluaciones en curso. Los registros por episodio aterrizan en una disposición autocontenida (ADR-eval-004) para que un compañero de equipo pueda reproducir cualquier episodio individual sin volver a ejecutar.

### 2.9 Observabilidad en Tiempo Real

Cada paso transmite observaciones, razonamiento, acciones y métricas vía WebSocket, enrutadas por `execution_id` para que las ejecuciones concurrentes no crucen sus flujos. Los errores de cualquier fuente — excepciones de nodos, caídas de subprocesos en modo servidor y fallos HTTP — fluyen a través de un `ErrorBus` unificado y aparecen como entradas en la pestaña Report + toasts (ADR-observability-004). (Los errores de renderizado de React son capturados por un error boundary del lado del cliente.)

---

## 3. Ruta de Sim a Real

AgentCanvas está diseñado para la portabilidad: un único grafo de agente puede ejecutarse contra un simulador hoy y migrar a un robot real en el futuro sin cambios a nivel de grafo. Esta propiedad se deriva de dos decisiones arquitectónicas — los entornos son ellos mismos nodesets (ADR-components-002), y cualquier nodeset puede ejecutarse en un runtime aislado mediante el *modo servidor* (ADR-server-001).

### Hoy: Nodesets de Simulador

Los entornos incluidos — Habitat (VLN-CE), MatterSim / MP3D, HM-EQA, OpenEQA, SIMPLER (VLA real-a-sim), y LIBERO (manipulación) — están cada uno implementados como un `BaseNodeSet` que expone puertos de observación y acción. El grafo de agente se conecta a estos puertos y nunca importa el simulador directamente, lo que mantiene el grafo independiente de cualquier implementación de entorno específica.

### Mañana: Un Nodeset de ROS con la Misma Interfaz

El despliegue en robot real se logra reemplazando el nodeset de simulador por un **nodeset de ROS** que expone la misma interfaz `observation` / `act`. Internamente, este nodeset compone componentes de ROS existentes — `cv_bridge`, `Nav2`, `MoveIt` y paquetes de drivers de hardware — en una fachada unificada. El modo servidor lanza el nodeset dentro de su propio entorno Python de ROS y lo conecta con el lienzo por HTTP. El grafo de agente en sí permanece sin cambios.

Esta división del trabajo es favorable porque la ingeniería sustancial — percepción, control, planificación de movimiento e interfaz con el hardware — ya existe como paquetes de ROS maduros. El adaptador del lado de ROS es por tanto una tarea de composición en lugar de desarrollo desde cero, y el nodeset de entorno del lado de AgentCanvas se reduce a un cliente HTTP ligero.

### Integración Bidireccional

El límite entre AgentCanvas y ROS es simétrico; cualquiera de los dos lados puede ser dueño del bucle de control:

- **ROS como subsistema de AgentCanvas** *(patrón nativo; el modo servidor está diseñado para este caso)* — el nodeset de ROS se ejecuta en modo servidor, AgentCanvas dirige el bucle del agente, y ROS proporciona el sensado y la actuación.
- **AgentCanvas como subsistema de ROS** *(también admitido; no requiere modificaciones del framework)* — cuando el proyecto más amplio está liderado por ROS, el bucle de control del lado de ROS invoca el endpoint `/run` de AgentCanvas en cada paso (tratando el grafo como una política) y publica la acción devuelta. Esto solo requiere un nodo puente de ROS ligero del lado de ROS.

### Visibilidad de las Dependencias de Ground-Truth

La misma abstracción de nodeset aborda directamente dos puntos débiles planteados en §1. Un nodo que consulta el ground truth del simulador (p. ej., `env_habitat__get_object_pose`) y un nodo que realiza percepción real (p. ej., un detector basado en SAM) aparecen como bloques visiblemente distintos en el lienzo. Que un agente dependa del ground truth o de la percepción es por tanto una propiedad de la topología del grafo, no un detalle de implementación oculto. Sustituir uno por el otro es un cambio de arista local, no una refactorización de código.

### Estado

Todos los nodesets de entorno incluidos actualmente están basados en simuladores. Un **nodeset de ROS para robot real sigue siendo una ranura [en busca de contribución](#8-contribuir)** — el camino arquitectónico está establecido y es intencional, y los componentes necesarios del lado de ROS ya están disponibles en el ecosistema.

---

## 4. Primeros Pasos

Hay dos formas de usar AgentCanvas, ambas sobre el mismo sustrato de grafo tipado:

1. **Construir y ejecutar un grafo a mano** — compón nodos en el lienzo, ejecuta un agente en vivo contra un simulador, y evalúalo a escala (el resto de esta sección).
2. **Búsqueda de Arquitectura de Agentes (AAS)** — entrega un grafo semilla a un agente de programación y deja que busque arquitecturas por ti ([saltar](#44-ejecutar-la-búsqueda-de-arquitectura-de-agentes-aas)).

### 4.1 Requisitos Previos

- Python 3.10+ con Conda (el entorno `agentcanvas` por defecto — ADR-platform-004)
- Node.js 18+
- *(Opcional, para Habitat-Sim)* un entorno Python 3.8 separado — `habitat-sim 0.1.7` solo se ejecuta aquí; AgentCanvas se comunica con él mediante el modo servidor, consulta [INSTALL.md](INSTALL.md)

### 4.2 Ejecutar el Panel Web

```bash
# Activar el entorno
conda activate agentcanvas

# Iniciar el backend (FastAPI :8000) + frontend (Vite :5173)
cd agentcanvas && bash run_dev.sh
```

Abre [http://localhost:5173](http://localhost:5173) para acceder al editor de lienzo.

### 4.3 Ejecutar una Evaluación

El mismo pipeline de evaluación se expone a través de cuatro interfaces — elige según lo que tengas a mano:

| # | Interfaz | Audiencia | Ideal para |
|---|-----------|----------|----------|
| 1 | **Página de Evaluación del frontend** | Humano                | Guiado por clics, observa el progreso en vivo en la UI |
| 2 | **Comando slash `/experiment:run`** | Agente de programación (Claude Code) | Admisión de GPU controlada por perfil, puerto auto-asignado, sin pisar `:8000` |
| 3 | **Servidor MCP** | Agente de programación              | Evaluación conversacional y ad-hoc — sin sobrecarga de comandos slash |
| 4 | **API HTTP** | Scripts / CI                | REST directo, sin necesidad de MCP |

#### 1. Página de Evaluación del frontend — para humanos

Abre un grafo guardado en la página **Eval**, elige un split + rango de episodios, pulsa **Start**. El progreso se transmite en vivo por WebSocket; los resultados aterrizan como JSONL por episodio bajo `outputs/eval_runs/{run_id}/episodes/ep{NNNN}/` (ADR-eval-004) y se pueden explorar en el panel Run Detail. El fan-out de entornos multi-worker y la inferencia por lotes son configurables desde el formulario (ADR-eval-002).

→ [Tutorial de Evaluación por Lotes](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/tutorials/batch-eval.html)

#### 2. `/experiment:run` — para agentes de programación en este repositorio

Al usar Claude Code, `/experiment:run <profile> -- <cmd>` envuelve cualquier invocación de evaluación en la compuerta de admisión del `JobScheduler` del backend (ADR-eval-003): el wrapper reclama VRAM bajo el perfil declarado en `.claude/commands/experiment/profiles.yaml`, lanza el backend en un puerto asignado (`BACKEND_URL=http://127.0.0.1:<port>` se exporta al comando envuelto), y libera la ranura al salir. Comandos complementarios: `/experiment:status` para instantáneas de ejecución, `/experiment:teardown` para una cancelación elegante.

→ [`.claude/commands/experiment/README.md`](../.claude/commands/experiment/README.md)

Para bucles completos de diseño de búsqueda de arquitectura (muchas iteraciones de proponer → evaluar → quedarse-con-el-mejor sobre un grafo semilla), consulta [Ejecutar la Búsqueda de Arquitectura de Agentes](#44-ejecutar-la-búsqueda-de-arquitectura-de-agentes-aas) más abajo.

#### 3. Servidor MCP — para agentes de programación

Registra `agentcanvas-backend` con cualquier cliente compatible con MCP (Claude Code, Cursor, …) y llama a herramientas tipadas (`graph_list`, `eval_start`, `eval_status`, `eval_export`, `eval_stop`) de forma conversacional. Sin contabilidad de árbol de iteraciones — solo evaluación directa contra un backend prestado-o-lanzado.

→ [`agentcanvas/mcp_server/README.md`](../agentcanvas/mcp_server/README.md)

#### 4. API HTTP — para scripts y CI

REST directo para scripts, CI o entornos sin MCP:

```bash
curl -X POST http://localhost:8000/api/eval/v2/start \
  -H 'content-type: application/json' \
  -d '{"graph_name": "navgpt_ce", "split": "val_unseen", "worker_count": 4}'
# sondear  GET /api/eval/v2/status
# obtener GET /api/eval/v2/export/{run_id}
```

→ [Controlar el Backend desde un Agente de Programación](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/tutorials/coding-agent-backend.html) — análisis profundo de todos los modos programáticos en paralelo

### 4.4 Ejecutar la Búsqueda de Arquitectura de Agentes (AAS)

Más allá de evaluar un grafo fijo, AgentCanvas es el sustrato para la **Búsqueda de Arquitectura de Agentes** — un bucle en tiempo de desarrollo donde un *Optimizer* agente-de-programación LLM propone repetidamente ediciones de grafo a un *Executor* semilla, evalúa cada candidato en el simulador, y conserva las mejoras (§1 — [por qué un sustrato buscable](#1-por-qué-agentcanvas)). Como un agente es un grafo tipado, cada candidato es un parche con tipos verificados que se ejecuta antes de cualquier rollout costoso, y los registros de episodios por nodo permiten al Optimizer atribuir los cambios de puntuación a módulos específicos.

<p align="center">
  <img src="../assets/readme/aas-search.gif" alt="El optimizador agente-de-programación buscando sobre el grafo de un ejecutor encarnado — proponiendo ediciones, ejecutándolas, conservando las ganancias" width="800">
  <br><sub><em>El optimizador agente-de-programación buscando sobre el grafo de un executor encarnado — proponer una edición, ejecutarla, conservar las ganancias.</em></sub>
</p>

La búsqueda está **sembrada por método**: `iter_0` es un método encarnado publicado y el bucle busca ediciones a nivel de grafo a su alrededor. Tres variantes de búsqueda se incluyen como skills de Claude Code bajo `.claude/commands/architect/`, compartiendo un único harness de agente-de-programación (proposer → implementer → evaluator) y diferenciándose solo en la lógica del proposer + la memoria persistente:

| Skill de variante | Nombre en el artículo | Política de búsqueda |
|---|---|---|
| `myloop` | **KDLoop** | Ciclo de cuatro fases THINK → CRITIC → EXPERIMENT → DISTILL, memoria tipada + meta-fase REFLECT |
| `adas-subagent` | **ADAS** (port) | Propuestas estilo Reflexion sobre un archivo plano de solo-anexado |
| `aflow` | **AFlow** (port) | Selección de padre por score-softmax + memoria anti-repetición |

```text
# En una sesión de Claude Code en este repositorio — ejecutar KDLoop sobre el ejecutor MapGPT
/architect:myloop:loop mapgpt_mp3d --goal "raise val_unseen SR"

# Los ports de ADAS / AFlow toman la misma  <graph> [<version>]  forma
/architect:adas-subagent:loop smartway_ce
/architect:aflow:loop explore_eqa_hmeqa
```

Grafos semilla actualmente cableados para búsqueda: `mapgpt_mp3d`, `smartway_ce` (VLN), `explore_eqa_hmeqa` (EQA), `voxposer_libero_monolithic` (VLA). Cada iteración escribe su propuesta, parche, puntuaciones de evaluación y registros bajo `outputs/design_runs/{variant}/{graph}/vN/iter_M/`.

→ [Referencia de pipelines de AAS](https://jianzhou0420.github.io/AgentCanvas/pages/aas/index.html)

### 4.5 Documentación

```bash
# Servir el doc-site localmente en :8092 (recarga en vivo vía SSE)
bash docs/run_dev.sh
```

---

## 5. Arquitectura

```
Frontend (React 18 + React Flow + Zustand)
    |
    |  REST + WebSocket
    v
Backend (FastAPI + Python 3.10+)
    |
    |-- WorkspaceComponentRegistry  -->  workspace/  (auto-descubrimiento)
    |-- GraphExecutor   -->  ejecución de grafos (DAG + cíclico + multi-ámbito)
    |-- AutoServerApp      -->  server-mode nodesets (entornos aislados)
    |-- HookRunner         -->  interceptores pre/post
    |-- JobScheduler       -->  admisión de evaluación subproceso-por-ejecución (ADR-eval-003)
    |-- ErrorBus           -->  reporte de errores unificado (ADR-observability-004)
    v
Simuladores (Habitat-Sim, MatterSim/MP3D, HM3D, SAPIEN/ManiSkill2, MuJoCo/robosuite, ...)
```

**Diseño clave**: El framework tiene **cero conocimiento de dominio** (ADR-platform-001). Todo el código específico de dominio — políticas VLN, prompts de LLM, herramientas de navegación, wrappers de entornos — vive en `workspace/`. El framework descubre componentes en tiempo de ejecución mediante herencia de clase base. Nunca importa código de dominio directamente; el límite de importación lo impone `agentcanvas/backend/app/test_import_boundary.py`.

---

## 6. Estructura del Proyecto

```
vlnworkspace/                  # raíz del repositorio (nombre heredado; la plataforma es "AgentCanvas")
├── agentcanvas/               # Aplicación web full-stack
│   ├── backend/app/         #   Backend FastAPI (motor de ejecución, APIs, servicios, errores)
│   ├── frontend/src/        #   React + TypeScript (editor de lienzo)
│   └── mcp_server/          #   Servidor MCP para integración con agentes de programación
├── workspace/                 # Espacio de trabajo del usuario — todos los componentes de dominio (auto-descubiertos)
│   ├── nodesets/            #   Nodesets por paleta: env / method / model / policy (+ common, _upstream)
│   ├── graphs/              #   Grafos de agente guardados (kind="graph")
│   ├── graph_nodes/         #   Nodos compuestos reutilizables (kind="node")
│   ├── nodes/               #   Subclases independientes de BaseCanvasNode
│   ├── architect/           #   Perfiles de búsqueda AAS + andamiaje de ejecución
│   └── hooks.json           #   Definiciones de hooks a nivel de workspace
├── data/                      # Datasets, pesos de modelos (gitignored)
├── outputs/                   # Salidas de evaluación + ejecuciones de diseño (eval_runs/, design_runs/, …)
├── docs/                      # Doc-site HTML escrito a mano (run_dev.sh → :8092)
├── third_party/               # Submódulos de Git (habitat-lab, VLN-CE, MatterSim, vla_workspace, …)
└── scripts/                   # Scripts de configuración de datos + instalación
```

---

## 7. Estado del Proyecto

AgentCanvas está **pre-1.0 y en desarrollo activo**. El estado se rastrea por versión, no por una lista de verificación de funcionalidades en curso — consulta la [Política de Versionado](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/repo/versioning.html) y [`major-versions.html`](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/core/major-versions.html) para más detalle.

- **v0.1 — experimentos AAS.** La instantánea sobre la que se ejecutaron las corridas de Búsqueda de Arquitectura de Agentes del artículo — un ancla de reproducibilidad para esos resultados, no una publicación pública.
- **v0.2 — vista previa de investigación (actual).** La primera publicación de código abierto: el editor de lienzo, el ejecutor de grafos (DAG + cíclico + multi-ámbito), los contenedores de estado, los nodesets en modo servidor auto-alojados, la evaluación por lotes, y más de 40 nodesets (env / method / model / policy) todos en producción. La API pública aún no está congelada, así que las publicaciones menores pueden romperla. Inventario incluido: [§2 Características](#2-características) y las páginas de estado de soporte [VLN](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/design-docs/vln-support-status.html) / [EQA](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/design-docs/eqa-support-status.html) / [VLA](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/design-docs/vla-support-status.html).
- **v1.0 — en progreso.** Se publicará cuando la API pública sea estable — de código abierto y congelada bajo SemVer, independiente de cualquier artículo.
- **v2.0 — futuro.** Ejecución que muta la topología: generación ilimitada de subagentes, fan-out en tiempo de ejecución sobre listas en tiempo de ejecución, nuevos tipos de herramientas emergiendo en tiempo de ejecución, grafos auto-modificables. Consulta [`major-versions.html`](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/core/major-versions.html) §2 para la tesis y las preguntas abiertas.

---

## 8. Contribuir

Dos tipos de contribución, ambos bienvenidos — consulta [CONTRIBUTING.md](../CONTRIBUTING.md):

- **Contenido — nodesets y grafos.** Escribe un nodeset que envuelva una herramienta / simulador / modelo (p. ej. 3D Gaussian Splatting en tiempo real, un sistema SLAM basado en vóxeles) o que codifique un método (p. ej. NavGPT, MapGPT), o compón un grafo que cablee nodesets existentes en un agente completo. Abre un PR a `workspace/`; la revisión es ligera.
- **Core — UI, backend, framework.** Correcciones de errores, nuevas funcionalidades, incluso refactorizaciones son bienvenidas. La única petición: si un cambio es lo suficientemente grande como para costar tiempo real, abre primero una [Discussion](https://github.com/jianzhou0420/AgentCanvas/discussions) para que podamos alinearnos antes de que construyas.

Cada nodeset y grafo se acredita a su autor/mantenedor en el tablero de Créditos de abajo — con un enlace de cita si tiene un artículo asociado — así que contribuir aquí no te cuesta la autoría.

### Créditos

<table>
<tr><th>Componente</th><th>Creado por</th></tr>
<tr>
<td><b>Framework AgentCanvas</b></td>
<td><a href="https://github.com/jianzhou0420">@jianzhou0420</a></td>
</tr>
<tr>
<td>

<details open>
<summary><b>Primera publicación</b> — nodesets incluidos, grafos de referencia, doc-site</summary>

<br>

<b>Simuladores / entornos</b>

- Habitat (navegación continua VLN-CE)
- Matterport3D / MatterSim (navegación panorámica discreta)
- HM-EQA (entorno de QA encarnado)
- OpenEQA (benchmark de QA encarnado, modo EM-EQA)
- SIMPLER (evaluación VLA real-a-sim de SAPIEN / ManiSkill2)
- LIBERO (manipulación MuJoCo / robosuite, 5 suites)

<b>Métodos de agente / razonamiento</b>

<i>EQA</i>

- Baselines de OpenEQA EM-EQA — blind-LLM / single-frame / multi-frame (`openeqa_em_*.json`) ✅ todos verificados; LLM-Match multi-frame 0.7025 vs 0.466 del artículo (el razonador+juez gpt-4o supera al gpt-4 / gpt-4-vision-preview del artículo)
- Explore-EQA (exploración de frontera fijada a Prismatic en HM-EQA) ✅ verificado — SR 0.42 reproduce el baseline de 0.44
- ToolEQA (solo HM-EQA — sustrato PortBench v1) — rehecho monolito-primero 2026-06-08; se ejecuta de extremo a extremo (ReAct + go_next con TSDF fusionado + Qwen2.5-VL/DetAny3D sobre HTTP en modo servidor), ajuste de SR en progreso

<i>VLN</i>

- NavGPT (primitivas de razonamiento pensamiento–acción con LLM) ✅ funciona con gpt-4 (caro); otros LLMs no probados (se sabe que gpt-4o regresa en prompts ReAct largos)
- MapGPT (agente LLM de topo-mapa lingüístico, ACL 2024) ✅ verificado — SR 0.477 / 0.463 en MapGPT_72
- SmartWay-mono (predictor de waypoints VLN-CE) ✅ comparable al artículo — SR 0.270 vs 0.29 del artículo
- SmartWay-CE ✅ condición de carrera de finalización silenciosa corregida; se ejecuta de extremo a extremo en evaluación de 20 workers
- SpatialNav (navegación por grafo espacial) ❌ sin verificar — SR=0
- Open-Nav (navegación de vocabulario abierto) ❌ sin verificar — SR=0
- DiscussNav (debate multi-LLM, fan-out acotado) ❓ en progreso — fitness aún no llevado a comparable con el artículo
- Three-Step Nav (navegación por waypoints zero-shot, subclase de Open-Nav) ❓ validado de extremo a extremo — SR 0.10 / oracle 0.30 @10ep; ajuste comparable al artículo pendiente
- AO-Planner (SAM + LLM + planificador de rutas 3D, AAAI 2025) ❓ en progreso — nodeset incluido, evaluación pendiente
- Basic Agent (kit de herramientas VLN fundamental — 11 nodos en 5 categorías)

<i>VLA</i>

- Los métodos específicos de VLA (Pi0 / SmolVLA / DP / DROID-DP / Octo / VoxPoser-LIBERO) viven bajo <b>Políticas</b> más abajo — tienen forma de política (observación-de-entorno → acción) en lugar de forma de razonamiento, así que se agrupan por estructura de código en lugar de por familia de tarea

<b>Percepción / visión</b>

- SAM (Segment Anything)
- BLIP-2 + Faster R-CNN (captioning y detección)
- RAM (modelo recognize-anything)
- SpatialBot (VLM consciente de la profundidad)
- Prismatic VLM (puntuación por verosimilitud de tokens + generación libre)
- Mapeo TSDF
- Grafo semántico de escena

<b>Políticas</b>

- CMA (baseline VLN-CE de Atención Cross-Modal) ✅ verificado — `straightforward.json` promovido a verified/, SR 0.38 / SPL 0.348, idéntico bit a bit al nativo
- Octo (generalista VLA, baseline nativo de SIMPLER) ✅ el baseline se ejecuta en `octo_simpler.json`
- Framework VLA genérico (adaptador Pi0 / SmolVLA / DP / DROID-DP) ✅ Pi0 verificado — 5/5 en `vla_policy_libero` libero_spatial task 0; variante SIMPLER por determinar
- VoxPoser-LIBERO (LMP + mapa-de-coste-por-vóxeles + OSC) ✅ verificado de extremo a extremo (agarre + transporte); SR registrado
- Adaptador de política VLN-CE (registro R2R-CE de 12 variantes — 2 publicadas upstream, 10 ablaciones marcadas como placeholder)

<b>Doc-site</b> — HTML escrito a mano (tras la retirada de MkDocs el 2026-05-18) con 46 ADRs, glosario, páginas de capacidades, tutoriales, documentos de diseño

</details>

</td>
<td><a href="https://github.com/jianzhou0420">@jianzhou0420</a></td>
</tr>
<tr>
<td><b>Benchmark:</b> AI2-THOR <i>(ALFRED / TEACh — E4)</i></td>
<td><i><a href="../CONTRIBUTING.md">en busca de contribución</a></i></td>
</tr>
<tr>
<td><b>Benchmark:</b> RxR-CE <i>(VLN-CE multilingüe — E2)</i></td>
<td><i><a href="../CONTRIBUTING.md">en busca de contribución</a></i></td>
</tr>
<tr>
<td><b>Benchmark:</b> REVERIE <i>(grounding de objetos remotos — E3)</i></td>
<td><i><a href="../CONTRIBUTING.md">en busca de contribución</a></i></td>
</tr>
<tr>
<td><b>Benchmark:</b> OpenEQA A-EQA <i>(modo EQA activo — E10)</i></td>
<td><i><a href="../CONTRIBUTING.md">en busca de contribución</a></i></td>
</tr>
<tr>
<td><b>Método:</b> HAMT <i>(transformer de historia jerárquica — M5)</i></td>
<td><i><a href="../CONTRIBUTING.md">en busca de contribución</a></i></td>
</tr>
<tr>
<td><b>Método:</b> DUET <i>(transformer de grafos de doble escala — M6)</i></td>
<td><i><a href="../CONTRIBUTING.md">en busca de contribución</a></i></td>
</tr>
<tr>
<td><b>Método:</b> MapGPT (variante de rejilla métrica) <i>(LLM + ocupación derivada de profundidad — M2; distinta de la variante topo-lingüística incluida)</i></td>
<td><i><a href="../CONTRIBUTING.md">en busca de contribución</a></i></td>
</tr>
<tr>
<td><b>Método:</b> InstructNav <i>(Dynamic CoN + Multi-Sourced Value Maps, CoRL 2024 — M8)</i></td>
<td><i><a href="../CONTRIBUTING.md">en busca de contribución</a></i></td>
</tr>
<tr>
<td><b>Método:</b> VLN-SIG <i>(grounding de sub-instrucciones — M4)</i></td>
<td><i><a href="../CONTRIBUTING.md">en busca de contribución</a></i></td>
</tr>
<tr>
<td><b>Funcionalidad:</b> Nodeset de memoria <i>(recuerdo episódico + búsqueda semántica — F1)</i></td>
<td><i><a href="../CONTRIBUTING.md">en busca de contribución</a></i></td>
</tr>
<tr>
<td><b>Funcionalidad:</b> Ejecución paralela de nodos <i>(modelo de superpaso Pregel — F3)</i></td>
<td><i><a href="../CONTRIBUTING.md">en busca de contribución</a></i></td>
</tr>
<tr>
<td><b>Funcionalidad:</b> Exportar grafo como Python independiente <i>(evaluación por lotes headless — F4)</i></td>
<td><i><a href="../CONTRIBUTING.md">en busca de contribución</a></i></td>
</tr>
<tr>
<td><b>Infra:</b> Modo servidor Docker <i>(contenedores Habitat / MP3D — F7)</i></td>
<td><i><a href="../CONTRIBUTING.md">en busca de contribución</a></i></td>
</tr>
<tr>
<td><b>Infra:</b> Nodeset de ROS <i>(despliegue en robot real mediante modo servidor — §3)</i></td>
<td><i><a href="../CONTRIBUTING.md">en busca de contribución</a></i></td>
</tr>
</table>


---

## 9. Citación

Si utilizas AgentCanvas en tu investigación, cítalo así:

```bibtex
@misc{jian2026AgentCanvas,
  title         = {Automating the Design of Embodied Agent Architectures},
  author        = {Jian Zhou and Sihao Lin and Jin Li and Shuai Fu and Gengze Zhou and Qi Wu},
  year          = {2026},
  eprint        = {2606.30111},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2606.30111}
}
```

---

## 10. Licencia

Licencia Apache 2.0 — consulta [LICENSE](../LICENSE).
