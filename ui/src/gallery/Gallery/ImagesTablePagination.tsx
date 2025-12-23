import {
  Pagination,
  PaginationContent,
  PaginationEllipsis,
  PaginationItem,
  PaginationLink,
  PaginationNext,
  PaginationPrevious,
} from "@/general/components/pagination";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/general/components/select";

export type PaginationProps = {
  page: number;
  itemsCount: number;
  onPageChange: (page: number) => void;
  itemsPerPage: number;
  onItemsPerPageChange: (items_per_page: number) => void;
  isPending?: boolean;
};

export const PAGE_SIZES = [12, 24, 48, 96] as const;

function buildPageItems(page: number, totalPages: number) {
  if (totalPages <= 7)
    return Array.from({ length: totalPages }, (_, i) => i + 1);

  const items: Array<number | null> = [1];

  const left = Math.max(2, page - 1);
  const right = Math.min(totalPages - 1, page + 1);

  if (left > 2) items.push(null);

  for (let page = left; page <= right; page++) items.push(page);

  if (right < totalPages - 1) items.push(null);

  items.push(totalPages);
  return items;
}

export function ImagesTablePagination({
  page,
  itemsCount,
  onPageChange,
  isPending,
  itemsPerPage,
  onItemsPerPageChange,
}: PaginationProps) {
  const canPrev = page > 1;
  const totalPages = Math.ceil(itemsCount / itemsPerPage);
  const canNext = page < totalPages;
  const pageItems = buildPageItems(page, totalPages);

  return (
    <div className="flex px-4 py-2">
      <Pagination>
        <PaginationContent>
          <PaginationItem>
            <PaginationPrevious
              href="#"
              aria-disabled={!canPrev}
              onClick={(event) => {
                event.preventDefault();
                if (canPrev) onPageChange(page - 1);
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
                  onClick={(event) => {
                    event.preventDefault();
                    if (!isPending) onPageChange(page);
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
                event.preventDefault();
                if (canNext) onPageChange(page + 1);
              }}
            />
          </PaginationItem>
        </PaginationContent>
      </Pagination>
      <Select
        onValueChange={(value) => onItemsPerPageChange(Number.parseInt(value))}
      >
        <SelectTrigger>
          <SelectValue placeholder={`Items per page: ${itemsPerPage}`} />
        </SelectTrigger>
        <SelectContent>
          {PAGE_SIZES.map((size) => (
            <SelectItem key={size} value={String(size)}>
              {`Items per page: ${size}`}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
}
