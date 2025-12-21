import { ThumbnailCell } from '@/gallery/Gallery/ThumbnailCell'
import { useListImages } from '@/gallery/hooks'
import { imageSearchQuerySchema, type ImageSearchQuery, type IndexedImage } from '@/gallery/schema'
import { Card, CardContent, CardHeader, CardTitle } from '@/general/components/card'
import { Pagination, PaginationContent, PaginationEllipsis, PaginationItem, PaginationLink, PaginationNext, PaginationPrevious } from '@/general/components/pagination'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/general/components/table'
import { flexRender, getCoreRowModel, useReactTable, type ColumnDef, type SortingState } from '@tanstack/react-table'
import { ArrowDownNarrowWide, ArrowUpNarrowWide } from 'lucide-react'
import { useState } from 'react'


function buildPageItems(page: number, totalPages: number) {
  if (totalPages <= 7) return Array.from({ length: totalPages }, (_, i) => i + 1)

  const items: Array<number | null> = [1]

  const left = Math.max(2, page - 1)
  const right = Math.min(totalPages - 1, page + 1)

  if (left > 2) items.push(null)

  for (let page = left; page <= right; page++) items.push(page)

  if (right < totalPages - 1) items.push(null)

  items.push(totalPages)
  return items
}

const columns: ColumnDef<IndexedImage>[] = [
  {
    id: "user_visible_name",
    header: "Image",
    accessorFn: (row) => row.user_visible_name,
    cell: ({row}) =>  <ThumbnailCell image={row.original} />,
  },
  {
    accessorKey: "created_at",
    header: "Created At",
    cell: (info) => {
      const v = info.getValue() as string | null
      return v ? new Date(v).toLocaleString() : "-"
    },
  },
  {
    accessorKey: "modified_at",
    header: "Modified At",
    cell: (info) => {
      const v = info.getValue() as string | null
      return v ? new Date(v).toLocaleString() : "-"
    },
  },
]

const NAME_CONTAINS = "" // will be there later :)
const ITEMS_PER_PAGE = 12

export function Gallery() {
  const [page, setPage] = useState(1)
  const [sorting, setSorting] = useState<SortingState>([
    { id: "user_visible_name", desc: true },
  ])
  const orderBy = (sorting[0]?.id ?? "modified_at") as ImageSearchQuery["order_by"]
  const direction = (sorting[0]?.desc ? "descending" : "ascending") as ImageSearchQuery["direction"]


  const query = imageSearchQuerySchema.parse({
      page,
      items_per_page: ITEMS_PER_PAGE,
      order_by: orderBy,
      direction,
      name_contains: NAME_CONTAINS.trim() ? NAME_CONTAINS.trim() : undefined,
    })

  const { data, isPending, isFetching } = useListImages(query)

  const images = data?.images;
  const total = data?.total;
  const totalPages = total !== undefined ? Math.ceil(total / ITEMS_PER_PAGE) : undefined;

  const canPrev = page > 1 && !isFetching;
  const canNext = total !== undefined ? page * ITEMS_PER_PAGE < total && !isFetching : true;

  const pageItems =
    totalPages !== undefined ? buildPageItems(page, totalPages) : [page]

  // eslint-disable-next-line react-hooks/incompatible-library
  const table = useReactTable({
    data: images ?? [],
    columns: columns,
    manualSorting: true,
    manualPagination: true,
    getCoreRowModel: getCoreRowModel(),
    onSortingChange: (updater) => {
      setPage(1);
      setSorting(prev => {
        console.log({prev, updater});
        if (typeof  updater === "function") {
          return updater(prev);
        }
        return updater;
      });
    },
    state: {
      sorting,
    },
  })

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>Images</CardTitle>
      </CardHeader>

      <CardContent className="space-y-4">
        <div className="rounded-md border">
          <Table className="table-fixed w-full">
            <TableHeader>
              {table.getHeaderGroups().map(headerGroup => (
                <TableRow key={headerGroup.id}>
                  {headerGroup.headers.map(header => (
                    <TableHead
                      key={header.id}
                      onClick={header.column.getToggleSortingHandler()}
                      className="cursor-pointer select-none"
                    >
                      <div className="flex items-center gap-2">
                      {flexRender(header.column.columnDef.header, header.getContext())}
                      {header.column.getIsSorted() === "asc" && <ArrowUpNarrowWide />}
                      {header.column.getIsSorted() === "desc" && <ArrowDownNarrowWide />}
                      </div>
                    </TableHead>
                  ))}
                </TableRow>
              ))}
            </TableHeader>

            <TableBody>
              {isPending ? (
                <TableRow>
                  <TableCell colSpan={columns.length}>
                    Loading...
                  </TableCell>
                </TableRow>
              ) : table.getRowModel().rows.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={columns.length}>
                    No images
                  </TableCell>
                </TableRow>
              ) : (
                table.getRowModel().rows.map(row => (
                  <TableRow key={row.id}>
                    {row.getVisibleCells().map(cell => (
                      <TableCell key={cell.id}>
                        {flexRender(cell.column.columnDef.cell, cell.getContext())}
                      </TableCell>
                    ))}
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </div>
        <Pagination>
          <PaginationContent>
            <PaginationItem>
              <PaginationPrevious
                href="#"
                aria-disabled={!canPrev}
                onClick={(event) => {
                  event.preventDefault()
                  if (canPrev) setPage((p) => Math.max(1, p - 1))
                }}
              />
            </PaginationItem>

            {pageItems.map((page, pageIdx) =>
              page === null ? (
                <PaginationItem key={`e-${pageIdx}`}>
                  <PaginationEllipsis />
                </PaginationItem>
              ) : (
                <PaginationItem key={page}>
                  <PaginationLink
                    href="#"
                    isActive={page === page}
                    onClick={(e) => {
                      e.preventDefault()
                      if (!isFetching) setPage(page)
                    }}
                  >
                    {page}
                  </PaginationLink>
                </PaginationItem>
              ),
            )}

            <PaginationItem>
              <PaginationNext
                href="#"
                aria-disabled={!canNext}
                onClick={(event) => {
                  event.preventDefault()
                  if (canNext) setPage((p) => p + 1)
                }}
              />
            </PaginationItem>
          </PaginationContent>
        </Pagination>
      </CardContent>
    </Card>
  )
}
