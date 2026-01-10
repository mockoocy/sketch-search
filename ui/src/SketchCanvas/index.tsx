import React, {
  forwardRef,
  useEffect,
  useImperativeHandle,
  useReducer,
  useRef,
  useState,
} from "react";

import {
  type Point,
  type Stroke,
  drawReducer,
} from "@/SketchCanvas/drawReducer";
import { useGalleryStore } from "@/store";

export type SketchCanvasHandle = {
  clear: () => void;
  undo: () => void;
  redo: () => void;
  exportPng: (blobCallback: (blob: Blob) => void) => void;
  isEmpty: () => boolean;
};

type SketchCanvasProps = {
  width: number;
  height: number;
  strokeWidth: number;
  maxHistory: number;
  onCommit: () => void;
};

function makePoint(
  event: React.PointerEvent<HTMLCanvasElement>,
  canvas: HTMLCanvasElement,
): Point {
  const boundingRect = canvas.getBoundingClientRect();
  const x = event.clientX - boundingRect.left;
  const y = event.clientY - boundingRect.top;
  return { x, y };
}

function fillBackground(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  color = "#fff",
) {
  ctx.save();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = color;
  ctx.fillRect(0, 0, width, height);
  ctx.restore();
}

function clearCanvas(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  background = "#fff",
) {
  ctx.clearRect(0, 0, width, height);
  fillBackground(ctx, width, height, background);
}

function applyStrokeStyle(ctx: CanvasRenderingContext2D, strokeWidth: number) {
  ctx.strokeStyle = "#000";
  ctx.fillStyle = "#000";
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.lineWidth = strokeWidth;
}

function drawDot(ctx: CanvasRenderingContext2D, p: Point, strokeWidth: number) {
  const r = strokeWidth / 2;
  ctx.beginPath();
  ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
  ctx.fill();
}

function drawSegment(ctx: CanvasRenderingContext2D, a: Point, b: Point) {
  applyStrokeStyle(ctx, ctx.lineWidth);
  ctx.beginPath();
  ctx.moveTo(a.x, a.y);
  ctx.lineTo(b.x, b.y);
  ctx.stroke();
}

function drawStroke(ctx: CanvasRenderingContext2D, stroke: Stroke) {
  const pts = stroke.points;
  if (pts.length === 0) return;

  applyStrokeStyle(ctx, stroke.width);

  if (pts.length === 1) {
    drawDot(ctx, pts[0], stroke.width);
    return;
  }

  ctx.beginPath();
  ctx.moveTo(pts[0].x, pts[0].y);
  for (let i = 1; i < pts.length; i++) {
    ctx.lineTo(pts[i].x, pts[i].y);
  }
  ctx.stroke();
}

function redrawAll(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  strokes: Stroke[],
) {
  clearCanvas(ctx, width, height);
  for (const s of strokes) drawStroke(ctx, s);
}

export const SketchCanvas = forwardRef<
  SketchCanvasHandle,
  Partial<SketchCanvasProps>
>(function SketchCanvas(
  { width = 240, height = 240, strokeWidth = 2, maxHistory = 200, onCommit },
  ref,
) {
  const [isFocused, setIsFocused] = useState(false);
  const [isDrawing, setIsDrawing] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const currentStrokeRef = useRef<Stroke | null>(null);

  const [state, dispatch] = useReducer(drawReducer, {
    history: [],
    redo: [],
  });

  const clearSource = useGalleryStore((state) => state.clearSource);

  useImperativeHandle(
    ref,
    () => ({
      clear: () => {
        dispatch({ type: "clear" });
        if (canvasRef.current !== null) {
          redrawAll(canvasRef.current!.getContext("2d")!, width, height, []);
        }
        onCommit?.();
        clearSource();
      },
      undo: () => {
        dispatch({ type: "undo" });
        if (canvasRef.current !== null) {
          redrawAll(
            canvasRef.current!.getContext("2d")!,
            width,
            height,
            state.history.slice(0, -1),
          );
        }
        onCommit?.();
      },
      redo: () => {
        dispatch({ type: "redo" });
        if (canvasRef.current !== null) {
          const newHistory =
            state.redo.length > 0
              ? [...state.history, state.redo[state.redo.length - 1]]
              : state.history;
          redrawAll(
            canvasRef.current!.getContext("2d")!,
            width,
            height,
            newHistory,
          );
        }
        onCommit?.();
      },
      exportPng: (blobCallback: (blob: Blob) => void) => {
        const canvas = canvasRef.current;
        if (!canvas) return null;
        canvas.toBlob((blob) => {
          if (!blob) return;
          blobCallback(blob);
        }, "image/png");
      },
      isEmpty: () => state.history.length === 0,
    }),
    [onCommit, height, state.history, state.redo, width, clearSource],
  );

  function onPointerDown(event: React.PointerEvent<HTMLCanvasElement>) {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;

    canvas.setPointerCapture(event.pointerId);
    setIsDrawing(true);

    const point = makePoint(event, canvas);
    currentStrokeRef.current = { points: [point], width: strokeWidth };
    applyStrokeStyle(ctx, strokeWidth);
  }

  function onPointerMove(event: React.PointerEvent<HTMLCanvasElement>) {
    if (!isDrawing) return;
    const canvas = canvasRef.current;
    const stroke = currentStrokeRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx || !stroke) return;

    const point = makePoint(event, canvas);
    const currentPoints = stroke.points;
    const lastPoint = currentPoints[currentPoints.length - 1];
    currentPoints.push(point);
    drawSegment(ctx, lastPoint, point);
  }

  function onPointerUp(event: React.PointerEvent<HTMLCanvasElement>) {
    const canvas = canvasRef.current;
    if (!canvas) return;

    if (!isDrawing) return;
    setIsDrawing(false);

    const stroke = currentStrokeRef.current;
    currentStrokeRef.current = null;

    canvas.releasePointerCapture(event.pointerId);

    if (!stroke || stroke.points.length === 0) return;

    dispatch({ type: "commit", stroke, maxHistory });
    onCommit?.();
  }

  function onKeyDown(event: React.KeyboardEvent<HTMLCanvasElement>) {
    if (!isFocused) return;
    if (event.key === "z" && (event.metaKey || event.ctrlKey)) {
      event.preventDefault();
      dispatch({ type: "undo" });
      onCommit?.();
    } else if (event.key === "y" && (event.metaKey || event.ctrlKey)) {
      event.preventDefault();
      dispatch({ type: "redo" });
      onCommit?.();
    }
  }

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.floor(width * dpr);
    canvas.height = Math.floor(height * dpr);
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }, [height, width]);

  return (
    <canvas
      ref={canvasRef}
      color="#fff"
      tabIndex={0}
      onKeyDown={onKeyDown}
      className="w-full h-full rounded-md border bg-white outline-none focus:ring-2"
      style={{ touchAction: "none" }}
      onFocus={() => setIsFocused(true)}
      onBlur={() => setIsFocused(false)}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={() => {
        setIsDrawing(false);
        currentStrokeRef.current = null;
      }}
    />
  );
});
