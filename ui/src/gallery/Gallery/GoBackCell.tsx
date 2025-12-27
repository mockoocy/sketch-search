import { useGalleryStore } from "@/store";
import { Undo2 } from "lucide-react";

type GoBackCellProps = {
  parentPath: string;
};

export function GoBackCell({ parentPath }: GoBackCellProps) {
  const setDirectory = useGalleryStore((state) => state.setDirectory);

  return (
    <div
      className="flex items-center gap-2 cursor-pointer text-muted-foreground hover:text-foreground"
      onClick={() => setDirectory(parentPath)}
    >
      <Undo2 />
    </div>
  );
}
