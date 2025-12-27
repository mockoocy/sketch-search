import { DirectoryCell } from "@/gallery/Gallery/DirectoryCell";
import { GoBackCell } from "@/gallery/Gallery/GoBackCell";
import { ImagesTablePagination } from "@/gallery/Gallery/ImagesTablePagination";
import { ThumbnailCell } from "@/gallery/Gallery/ThumbnailCell";
import type { GalleryRow } from "@/gallery/hooks";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/general/components/table";
import {
  flexRender,
  getCoreRowModel,
  useReactTable,
  type ColumnDef,
  type OnChangeFn,
  type SortingState,
} from "@tanstack/react-table";
import { ArrowDownNarrowWide, ArrowUpNarrowWide } from "lucide-react";

const columns: ColumnDef<GalleryRow>[] = [
  {
    id: "user_visible_name",
    header: "Image",
    accessorFn: (row) => {
      if (row.kind === "directory") {
        return row.directory.path;
      } else if (row.kind === "image") {
        return row.image.user_visible_name;
      } else {
        return "..";
      }
    },
    cell: ({ row }) => {
      if (row.original.kind === "directory") {
        return <DirectoryCell row={row.original} />;
      } else if (row.original.kind === "image") {
        return <ThumbnailCell row={row.original} />;
      } else {
        return <GoBackCell parentPath={row.original.parentDirectory.path} />;
      }
    },
  },
  {
    accessorFn: (row) => {
      if (row.kind === "directory") {
        return row.directory.created_at;
      } else if (row.kind === "image") {
        return row.image.created_at;
      } else {
        return row.parentDirectory.created_at;
      }
    },
    header: "Created At",
    cell: (info) => {
      const value = info.getValue() as string | null;
      return value ? new Date(value).toLocaleString() : "-";
    },
  },
  {
    accessorFn: (row) => {
      if (row.kind === "directory") {
        return row.directory.modified_at;
      } else if (row.kind === "image") {
        return row.image.modified_at;
      } else {
        return row.parentDirectory.modified_at;
      }
    },
    header: "Modified At",
    cell: (info) => {
      const value = info.getValue() as string | null;
      return value ? new Date(value).toLocaleString() : "-";
    },
  },
];

type ImagesTableProps = {
  rows: GalleryRow[];
  gallerySize: number;
  onSortingChange: OnChangeFn<SortingState>;
  sorting: SortingState;
};

export function ImagesTable({
  rows,
  gallerySize,
  onSortingChange,
  sorting,
}: ImagesTableProps) {
  // eslint-disable-next-line react-hooks/incompatible-library
  const table = useReactTable({
    data: rows,
    columns: columns,
    manualSorting: true,
    manualPagination: true,
    getCoreRowModel: getCoreRowModel(),
    onSortingChange,
    state: { sorting },
  });

  return (
    <div className="rounded-md border">
      <Table className="table-fixed w-full border-b">
        <TableHeader>
          {table.getHeaderGroups().map((headerGroup) => (
            <TableRow key={headerGroup.id}>
              {headerGroup.headers.map((header) => (
                <TableHead
                  key={header.id}
                  onClick={header.column.getToggleSortingHandler()}
                  className="cursor-pointer select-none"
                >
                  <div className="flex items-center gap-2">
                    {flexRender(
                      header.column.columnDef.header,
                      header.getContext(),
                    )}
                    {header.column.getIsSorted() === "asc" && (
                      <ArrowUpNarrowWide />
                    )}
                    {header.column.getIsSorted() === "desc" && (
                      <ArrowDownNarrowWide />
                    )}
                  </div>
                </TableHead>
              ))}
            </TableRow>
          ))}
        </TableHeader>

        <TableBody>
          {table.getRowModel().rows.length === 0 ? (
            <TableRow>
              <TableCell colSpan={columns.length}>No images</TableCell>
            </TableRow>
          ) : (
            table.getRowModel().rows.map((row) => (
              <TableRow key={row.id}>
                {row.getVisibleCells().map((cell) => (
                  <TableCell key={cell.id}>
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </TableCell>
                ))}
              </TableRow>
            ))
          )}
        </TableBody>
      </Table>
      <ImagesTablePagination itemsCount={gallerySize} />
    </div>
  );
}
