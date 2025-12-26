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
  const [isOpen, setIsOpen] = useState(false);

  if (isPending) {
    return <Button disabled>Uploading...</Button>;
  }
  return (
    <Dialog
      open={isOpen}
      onOpenChange={(nextOpen) => {
        setIsOpen(nextOpen);
        if (!nextOpen) {
          setFile(null);
        }
      }}
    >
      <DialogTrigger asChild>
        <Button>Add image</Button>
      </DialogTrigger>

      <DialogContent>
        {file ? (
          <img
            src={URL.createObjectURL(file)}
            alt="Preview"
            style={{ maxWidth: "100%", marginBottom: "1rem" }}
          />
        ) : (
          <Input
            type="file"
            accept="image/*"
            onChange={(event) => setFile(event.target.files?.[0] ?? null)}
          />
        )}

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
