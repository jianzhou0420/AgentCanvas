/** Thin CodeMirror 6 wrapper for the source-editor drawer.
 *
 * One EditorView per mount. The document is replaced only when `docKey`
 * changes (file switch / fresh disk load) — never per keystroke — so the
 * cursor and undo history survive typing and parent re-renders.
 */

import { forwardRef, useEffect, useImperativeHandle, useRef } from "react";
import { Prec } from "@codemirror/state";
import { EditorView, keymap } from "@codemirror/view";
import { basicSetup } from "codemirror";
import { python } from "@codemirror/lang-python";
import { oneDark } from "@codemirror/theme-one-dark";

export interface CodeMirrorEditorHandle {
  /** Move the cursor to `class <name>` and scroll it into view (best-effort). */
  scrollToClass: (name: string) => void;
}

interface Props {
  /** Content as loaded from disk; applied only when `docKey` changes. */
  value: string;
  /** Identity of the loaded document — bump to replace the doc. */
  docKey: string;
  onChange: (text: string) => void;
  /** Ctrl/Cmd+S handler. */
  onSave?: () => void;
  /** true (default): fill the parent's height with an inner scroller.
   *  false: grow with content (stacked segment editors). Fixed at mount. */
  fill?: boolean;
}

const CodeMirrorEditor = forwardRef<CodeMirrorEditorHandle, Props>(
  function CodeMirrorEditor({ value, docKey, onChange, onSave, fill = true }, ref) {
    const hostRef = useRef<HTMLDivElement | null>(null);
    const viewRef = useRef<EditorView | null>(null);
    const valueRef = useRef(value);
    const onChangeRef = useRef(onChange);
    const onSaveRef = useRef(onSave);
    const fillRef = useRef(fill);
    valueRef.current = value;
    onChangeRef.current = onChange;
    onSaveRef.current = onSave;

    useEffect(() => {
      const view = new EditorView({
        doc: valueRef.current,
        extensions: [
          basicSetup,
          python(),
          oneDark,
          Prec.high(
            keymap.of([
              {
                key: "Mod-s",
                run: () => {
                  onSaveRef.current?.();
                  return true;
                },
              },
            ]),
          ),
          EditorView.updateListener.of((u) => {
            if (u.docChanged) onChangeRef.current(u.state.doc.toString());
          }),
          EditorView.theme(
            fillRef.current
              ? {
                  "&": { height: "100%", fontSize: "12px" },
                  ".cm-scroller": { overflow: "auto" },
                }
              : { "&": { fontSize: "12px" } },
          ),
        ],
        parent: hostRef.current!,
      });
      viewRef.current = view;
      return () => {
        view.destroy();
        viewRef.current = null;
      };
    }, []);

    const mountedKeyRef = useRef(docKey);
    useEffect(() => {
      if (mountedKeyRef.current === docKey) return; // initial doc set at mount
      mountedKeyRef.current = docKey;
      const view = viewRef.current;
      if (!view) return;
      view.dispatch({
        changes: { from: 0, to: view.state.doc.length, insert: valueRef.current },
      });
    }, [docKey]);

    useImperativeHandle(
      ref,
      () => ({
        scrollToClass: (name: string) => {
          const view = viewRef.current;
          if (!view) return;
          const idx = view.state.doc.toString().indexOf(`class ${name}`);
          if (idx < 0) return;
          view.dispatch({
            selection: { anchor: idx },
            effects: EditorView.scrollIntoView(idx, { y: "start", yMargin: 48 }),
          });
        },
      }),
      [],
    );

    return (
      <div
        ref={hostRef}
        className={fill ? "h-full min-h-0 overflow-hidden" : ""}
      />
    );
  },
);

export default CodeMirrorEditor;
