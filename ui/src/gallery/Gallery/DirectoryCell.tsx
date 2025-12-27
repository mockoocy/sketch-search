import type { DirectoryRow } from "@/gallery/hooks";
import { useGalleryStore } from "@/store";
import { Folder } from "lucide-react";

type DirectoryCellProps = {
  row: DirectoryRow;
};

export function DirectoryCell({ row }: DirectoryCellProps) {
  const { path } = row.directory;
  const lastDirectory = path.split("/").filter(Boolean).at(-1);
  const setDirectory = useGalleryStore((state) => state.setDirectory);
  return (
    <div className="flex items-center gap-2" onClick={() => setDirectory(path)}>
      <Folder className="h-8 w-8" />
      <span className="text-lg">{lastDirectory}</span>
    </div>
  );
}
