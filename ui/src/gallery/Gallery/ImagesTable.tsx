import { ImagesTablePagination } from "@/gallery/Gallery/ImagesTablePagination";
import { ThumbnailCell } from "@/gallery/Gallery/ThumbnailCell";
import { useListImages } from "@/gallery/hooks";
import type { ImageSearchQuery, IndexedImage } from "@/gallery/schema";
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

const columns: ColumnDef<IndexedImage>[] = [
  {
    id: "user_visible_name",
    header: "Image",
    accessorFn: (row) => row.user_visible_name,
    cell: ({ row }) => <ThumbnailCell image={row.original} />,
  },
  {
    accessorKey: "created_at",
    header: "Created At",
    cell: (info) => {
      const v = info.getValue() as string | null;
      return v ? new Date(v).toLocaleString() : "-";
    },
  },
  {
    accessorKey: "modified_at",
    header: "Modified At",
    cell: (info) => {
      const v = info.getValue() as string | null;
      return v ? new Date(v).toLocaleString() : "-";
    },
  },
];

type ImagesTableProps = {
  query: ImageSearchQuery;
  onSortingChange: OnChangeFn<SortingState>;
  sorting: SortingState;
  onPageChange: (page: number) => void;
  onItemsPerPageChange: (items_per_page: number) => void;
};

export function ImagesTable({
  query,
  onSortingChange,
  sorting,
  onPageChange,
  onItemsPerPageChange,
}: ImagesTableProps) {
  const { data, isPending } = useListImages(query);
  const images = data?.images ?? [];
  const total = data?.total ?? 0;
  // eslint-disable-next-line react-hooks/incompatible-library
  const table = useReactTable({
    data: images ?? [],
    columns: columns,
    manualSorting: true,
    manualPagination: true,
    getCoreRowModel: getCoreRowModel(),
    onSortingChange,
    state: { sorting },
  });

  return (
    <div className="rounded-md border">
      <Table className="table-fixed w-full">
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
          {isPending ? (
            <TableRow>
              <TableCell colSpan={columns.length}>Loading...</TableCell>
            </TableRow>
          ) : table.getRowModel().rows.length === 0 ? (
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
      <ImagesTablePagination
        page={query.page}
        itemsCount={total}
        itemsPerPage={query.items_per_page}
        onPageChange={onPageChange}
        onItemsPerPageChange={onItemsPerPageChange}
        isPending={isPending}
      />
    </div>
  );
}
