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

export type TablePaginationProps = {
  itemsCount: number;
  page: number;
  itemsPerPage: number;
  onPageChange: (page: number) => void;
  onItemsPerPageChange: (itemsPerPage: number) => void;
  pageSizes?: readonly number[];
};

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

export function TablePagination({
  itemsCount,
  page,
  itemsPerPage,
  onPageChange,
  onItemsPerPageChange,
  pageSizes = [12, 24, 48, 96] as const,
}: TablePaginationProps) {
  const totalPages = Math.max(1, Math.ceil(itemsCount / itemsPerPage));
  const canPrev = page > 1;
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

          {pageItems.map((p, idx) =>
            p === null ? (
              <PaginationItem key={`e-${idx}`}>
                <PaginationEllipsis />
              </PaginationItem>
            ) : (
              <PaginationItem key={p}>
                <PaginationLink
                  href="#"
                  isActive={p === page}
                  onClick={(event) => {
                    event.preventDefault();
                    onPageChange(p);
                  }}
                >
                  {p}
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
        value={String(itemsPerPage)}
        onValueChange={(value) => onItemsPerPageChange(Number.parseInt(value))}
      >
        <SelectTrigger>
          <SelectValue placeholder={`Items per page: ${itemsPerPage}`} />
        </SelectTrigger>
        <SelectContent>
          {pageSizes.map((size) => (
            <SelectItem key={size} value={String(size)}>
              {`Items per page: ${size}`}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
}
