import {
  ToggleGroup,
  ToggleGroupItem,
} from "@/general/components/toggle-group";
import { useGalleryStore } from "@/store";

export function DirectoryScopeToggle() {
  const directory = useGalleryStore((state) => state.query.directory);
  const setDirectory = useGalleryStore((state) => state.setDirectory);

  const value = directory === null ? "global" : "local";

  return (
    <ToggleGroup
      type="single"
      value={value}
      onValueChange={(next) => {
        if (next === "global") {
          setDirectory(null);
        } else {
          setDirectory("."); // keep current directory
        }
      }}
    >
      <ToggleGroupItem value="local">This folder</ToggleGroupItem>
      <ToggleGroupItem value="global">All folders</ToggleGroupItem>
    </ToggleGroup>
  );
}
