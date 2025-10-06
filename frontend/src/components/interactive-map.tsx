"use client";

import type React from "react";
import { useEffect, useRef, useState, useCallback } from "react";
import { ZoomIn, ZoomOut } from "lucide-react";

import { Button } from "./ui/button";
//import { Input } from "./ui/input"

interface InteractiveMapProps {
  latitude: number;
  longitude: number;
  onLocationChange: (lat: number, lng: number) => void;
  expandedMode?: boolean;
}

export function InteractiveMap({
  latitude,
  longitude,
  onLocationChange,
  expandedMode = false,
}: InteractiveMapProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);

  const [zoom, setZoom] = useState(5);
  const [canvasDimensions, setCanvasDimensions] = useState({
    width: 800,
    height: 400,
  });

  useEffect(() => {
    const updateCanvasSize = () => {
      if (containerRef.current) {
        const width = containerRef.current.clientWidth;
        const height = expandedMode
          ? Math.min(containerRef.current.clientHeight, width * 0.75)
          : Math.min(500, Math.max(350, width * 0.6));
        setCanvasDimensions({ width, height });
      }
    };
    updateCanvasSize();
    window.addEventListener("resize", updateCanvasSize);
    return () => window.removeEventListener("resize", updateCanvasSize);
  }, [expandedMode]);

  useEffect(() => {
    drawMap();
  }, [zoom, latitude, longitude, canvasDimensions, drawMap]);

  const drawMap = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const { width, height } = canvasDimensions;

    ctx.fillStyle = "hsl(var(--muted))";
    ctx.fillRect(0, 0, width, height);

    const centerLat = latitude;
    const centerLon = longitude;
    const tileSize = 256;

    const scale = Math.pow(2, zoom);
    const centerTileX = ((centerLon + 180) / 360) * scale;
    const centerTileY =
      ((1 -
        Math.log(
          Math.tan((centerLat * Math.PI) / 180) +
            1 / Math.cos((centerLat * Math.PI) / 180)
        ) /
          Math.PI) /
        2) *
      scale;

    const tilesWide = Math.ceil(width / tileSize) + 2;
    const tilesHigh = Math.ceil(height / tileSize) + 2;

    const startTileX = Math.floor(centerTileX - tilesWide / 2);
    const startTileY = Math.floor(centerTileY - tilesHigh / 2);

    const offsetX = (centerTileX - Math.floor(centerTileX)) * tileSize;
    const offsetY = (centerTileY - Math.floor(centerTileY)) * tileSize;

    for (let x = 0; x < tilesWide; x++) {
      for (let y = 0; y < tilesHigh; y++) {
        const tileX = startTileX + x;
        const tileY = startTileY + y;

        if (tileX < 0 || tileY < 0 || tileX >= scale || tileY >= scale)
          continue;

        const img = new Image();
        img.crossOrigin = "anonymous";
        img.onload = () => {
          const drawX =
            x * tileSize - offsetX + width / 2 - (tilesWide * tileSize) / 2;
          const drawY =
            y * tileSize - offsetY + height / 2 - (tilesHigh * tileSize) / 2;
          ctx.drawImage(img, drawX, drawY, tileSize, tileSize);

          drawMarker(ctx, width, height);
        };
        img.src = `https://tile.openstreetmap.org/${zoom}/${tileX}/${tileY}.png`;
      }
    }

    drawMarker(ctx, width, height);
  }, [zoom, latitude, longitude, canvasDimensions]);

  const drawMarker = (
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number
  ) => {
    const markerX = width / 2;
    const markerY = height / 2;

    ctx.fillStyle = "rgba(0, 0, 0, 0.4)";
    ctx.beginPath();
    ctx.arc(markerX, markerY + 2, 14, 0, Math.PI * 2);
    ctx.fill();

    ctx.strokeStyle = "white";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.arc(markerX, markerY, 12, 0, Math.PI * 2);
    ctx.stroke();

    ctx.fillStyle = "hsl(var(--primary))";
    ctx.beginPath();
    ctx.arc(markerX, markerY, 9, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = "white";
    ctx.beginPath();
    ctx.arc(markerX, markerY, 3, 0, Math.PI * 2);
    ctx.fill();
  };

  const handleCanvasClick = (e: React.MouseEvent) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const { width, height } = canvasDimensions;

    const pixelOffsetX = x - width / 2;
    const pixelOffsetY = y - height / 2;

    const scale = Math.pow(2, zoom);
    const tileSize = 256;

    const centerTileX = ((longitude + 180) / 360) * scale;
    const centerTileY =
      ((1 -
        Math.log(
          Math.tan((latitude * Math.PI) / 180) +
            1 / Math.cos((latitude * Math.PI) / 180)
        ) /
          Math.PI) /
        2) *
      scale;

    const clickedTileX = centerTileX + pixelOffsetX / tileSize;
    const clickedTileY = centerTileY + pixelOffsetY / tileSize;

    const clickedLon = (clickedTileX / scale) * 360 - 180;
    const n = Math.PI - (2 * Math.PI * clickedTileY) / scale;
    const clickedLat =
      (180 / Math.PI) * Math.atan(0.5 * (Math.exp(n) - Math.exp(-n)));

    const clampedLat = Math.max(25.0625, Math.min(52.9375, clickedLat));
    const clampedLng = Math.max(-124.9375, Math.min(-67.0625, clickedLon));

    onLocationChange(clampedLat, clampedLng);
  };

  return (
    <div
      ref={containerRef}
      className={`w-full ${expandedMode ? "h-full" : ""}`}
    >
      <canvas
        ref={canvasRef}
        width={canvasDimensions.width}
        height={canvasDimensions.height}
        className="w-full cursor-crosshair rounded-lg border-2 border-primary/20 bg-muted shadow-lg transition-all hover:border-primary/40 hover:shadow-xl"
        onClick={handleCanvasClick}
        aria-label="Interactive map canvas"
        role="img"
      />

      <div className="mt-3 flex flex-wrap items-center gap-2">
        <Button
          variant="outline"
          size="sm"
          onClick={() => setZoom(Math.min(10, zoom + 1))}
        >
          <ZoomIn className="mr-1 h-4 w-4" />
          Zoom In
        </Button>
        <Button
          variant="outline"
          size="sm"
          onClick={() => setZoom(Math.max(3, zoom - 1))}
        >
          <ZoomOut className="mr-1 h-4 w-4" />
          Zoom Out
        </Button>
        <span className="ml-auto text-xs text-muted-foreground">
          Zoom: {zoom}
        </span>
      </div>
    </div>
  );
}
