import { useAddImage } from "@/gallery/hooks";
import { Button } from "@/general/components/button";
import {
  Dialog,
  DialogContent,
  DialogTrigger,
} from "@/general/components/dialog";
import { Input } from "@/general/components/input";
import { useState } from "react";

export function AddImageDialog() {
  const { mutate: add, isPending } = useAddImage();
  const [file, setFile] = useState<File | null>(null);

  if (isPending) {
    return <Button disabled>Uploading...</Button>;
  }
  return (
    <Dialog>
      <DialogTrigger asChild>
        <Button>Add image</Button>
      </DialogTrigger>

      <DialogContent>
        <Input
          type="file"
          accept="image/*"
          onChange={(event) => setFile(event.target.files?.[0] ?? null)}
        />

        <Button
          disabled={!file}
          onClick={() =>
            file &&
            add({
              file,
              directory: "",
            })
          }
        >
          Upload
        </Button>
      </DialogContent>
    </Dialog>
  );
}
