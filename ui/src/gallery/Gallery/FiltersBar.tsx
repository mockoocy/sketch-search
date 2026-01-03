import { AddImageDialog } from "@/gallery/Gallery/AddImageDialog";
import { DirectoryScopeToggle } from "@/gallery/Gallery/DirectoryScopeToggle";
import {
  ImageFilterPreview,
  type ImageFilterPreviewProps,
} from "@/gallery/Gallery/ImageFilterPreview";
import type { Filters } from "@/gallery/schema";
import { Button } from "@/general/components/button";
import { DatetimePicker } from "@/general/components/datetime-picker";
import { Input } from "@/general/components/input";
import { SketchSearchDialog } from "@/SketchCanvas/SketchSearchDialog";
import { useGalleryStore, type SimilaritySource } from "@/store";
import { Controller, useForm } from "react-hook-form";
type FiltersFormValues = {
  name_contains?: string;
  created_min?: Date;
  created_max?: Date;
  modified_min?: Date;
  modified_max?: Date;
};

type FiltersBarProps = {
  onSubmit: (filters: Filters) => void;
};

function toIso(date?: Date) {
  return date ? date.toISOString() : undefined;
}

function propsForSimilarityFilter(
  similaritySource: SimilaritySource,
): ImageFilterPreviewProps | null {
  if (!similaritySource) {
    return null;
  }
  if (similaritySource.kind === "sketch") {
    return { variant: "sketch", sketch: similaritySource.blob };
  } else {
    return { variant: "image", imageId: similaritySource.imageId };
  }
}

export function FiltersBar({ onSubmit }: FiltersBarProps) {
  const similaritySource = useGalleryStore((state) => state.similaritySource);
  const similarityFilterProps = propsForSimilarityFilter(similaritySource);
  const { register, control, reset, handleSubmit } = useForm<FiltersFormValues>(
    {
      mode: "onChange",
      defaultValues: {
        name_contains: "",
      },
    },
  );

  const onValidSubmit = (values: FiltersFormValues) => {
    onSubmit({
      name_contains: values.name_contains,
      created_min: toIso(values.created_min),
      created_max: toIso(values.created_max),
      modified_min: toIso(values.modified_min),
      modified_max: toIso(values.modified_max),
    });
  };

  return (
    <form
      className="w-full grid grid-cols-[auto_auto_auto] items-end gap-6 mb-4"
      onSubmit={handleSubmit(onValidSubmit)}
    >
      <div className="flex flex-col gap-2">
        <div className="text-sm">Name contains</div>
        <Input
          {...register("name_contains")}
          placeholder="e.g. cat"
          className="w-56"
        />
        <DirectoryScopeToggle />
      </div>

      <div className="grid grid-rows-2 gap-4">
        <div className="grid grid-cols-[8ch_1fr_1fr] items-center gap-3">
          <div className="text-sm text-muted-foreground">Created</div>
          <Controller
            control={control}
            name="created_min"
            render={({ field }) => (
              <DatetimePicker date={field.value} onChange={field.onChange} />
            )}
          />
          <Controller
            control={control}
            name="created_max"
            render={({ field }) => (
              <DatetimePicker date={field.value} onChange={field.onChange} />
            )}
          />
        </div>

        <div className="grid grid-cols-[8ch_1fr_1fr] items-center gap-3">
          <div className="text-sm text-muted-foreground">Modified</div>
          <Controller
            control={control}
            name="modified_min"
            render={({ field }) => (
              <DatetimePicker date={field.value} onChange={field.onChange} />
            )}
          />
          <Controller
            control={control}
            name="modified_max"
            render={({ field }) => (
              <DatetimePicker date={field.value} onChange={field.onChange} />
            )}
          />
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-2 justify-end">
        {similarityFilterProps && (
          <div className="flex flex-col gap-2">
            <span className="text-muted-foreground">Similar to:</span>
            <ImageFilterPreview {...similarityFilterProps} />
          </div>
        )}
        <div className="flex flex-col gap-4">
          <Button
            type="button"
            variant="secondary"
            onClick={() =>
              reset({
                name_contains: "",
                created_min: undefined,
                created_max: undefined,
                modified_min: undefined,
                modified_max: undefined,
              })
            }
          >
            Reset filters
          </Button>
          <Button type="submit">Apply filters</Button>
        </div>

        <div className="flex flex-col gap-4">
          <SketchSearchDialog />
          <AddImageDialog />
        </div>
      </div>
    </form>
  );
}
