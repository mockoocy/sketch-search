import { ConfirmDeleteDialog } from "@/admin/AdminPage/confirm-delete-dialog";
import { UserFormDialog } from "@/admin/AdminPage/user-form-dialog";
import {
  useAddUser,
  useDeleteUser,
  useEditUser,
  useSearchUsers,
} from "@/admin/hooks";
import type { User, UserRole, UserSearchQuery } from "@/admin/schema";
import { Button } from "@/general/components/button";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/general/components/card";
import { Input } from "@/general/components/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/general/components/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/general/components/table";
import { TablePagination } from "@/general/components/table-pagination";
import {
  flexRender,
  getCoreRowModel,
  useReactTable,
  type ColumnDef,
  type PaginationState,
} from "@tanstack/react-table";
import { Pencil, Plus, Trash2 } from "lucide-react";
import { useState } from "react";

type RoleFilter = "all" | UserRole;

export function AdminPage() {
  const [emailInput, setEmailInput] = useState("");
  const [emailQuery, setEmailQuery] = useState<string | undefined>(undefined);
  const [roleFilter, setRoleFilter] = useState<RoleFilter>("all");
  const [selectedUser, setSelectedUser] = useState<User | null>(null);
  const [email, setEmail] = useState<string>("");
  const [role, setRole] = useState<UserRole>("user");
  const [openDialog, setOpenDialog] = useState<
    "add" | "edit" | "delete" | null
  >(null);
  const [pagination, setPagination] = useState<PaginationState>({
    pageIndex: 0,
    pageSize: 24,
  });

  const query: UserSearchQuery = {
    email: emailQuery,
    role: roleFilter === "all" ? undefined : roleFilter,
    page: pagination.pageIndex + 1,
    pageSize: pagination.pageSize,
  };

  const { data, isLoading, isError, error } = useSearchUsers(query);

  const handleUserMutation = () => {
    setSelectedUser(null);
    setOpenDialog(null);
  };

  const addUser = useAddUser(handleUserMutation);
  const editUser = useEditUser(handleUserMutation);
  const deleteUser = useDeleteUser(handleUserMutation);
  const users = data?.users ?? [];
  const total = data?.total ?? 0;

  const pageCount = Math.max(1, Math.ceil(total / pagination.pageSize));

  const columns: ColumnDef<User>[] = [
    {
      accessorKey: "email",
      header: "Email",
      cell: (info) => String(info.getValue() ?? ""),
    },
    {
      accessorKey: "role",
      header: "Role",
      cell: (info) => String(info.getValue() ?? ""),
    },
    {
      id: "actions",
      header: "Actions",
      cell: ({ row }) => (
        <div className="flex items-center gap-2">
          <Button
            type="button"
            className="inline-flex h-8 w-8 items-center justify-center rounded-md border bg-muted/50 hover:bg-muted"
            onClick={() => {
              setSelectedUser(row.original);
              setOpenDialog("edit");
            }}
            aria-label="Edit user"
          >
            <Pencil className="h-4 w-4" />
          </Button>

          <Button
            type="button"
            className="inline-flex h-8 w-8 items-center justify-center rounded-md border bg-destructive/10 hover:bg-destructive/30"
            onClick={() => {
              setSelectedUser(row.original);
              setOpenDialog("delete");
            }}
            aria-label="Delete user"
          >
            <Trash2 className="h-4 w-4 text-destructive" />
          </Button>
        </div>
      ),
    },
  ];
  // eslint-disable-next-line react-hooks/incompatible-library
  const table = useReactTable({
    data: users,
    columns,
    getCoreRowModel: getCoreRowModel(),
    manualPagination: true,
    pageCount,
    state: { pagination },
    onPaginationChange: setPagination,
  });

  const applyFilters = () => {
    const trimmed = emailInput.trim();
    setEmailQuery(trimmed.length ? trimmed : undefined);
    setPagination((p) => ({ ...p, pageIndex: 0 }));
  };

  const onRoleChange = (value: RoleFilter) => {
    setRoleFilter(value);
    setPagination((p) => ({ ...p, pageIndex: 0 }));
  };

  return (
    <Card className="w-full">
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle>Users</CardTitle>
        <Button type="button" onClick={() => setOpenDialog("add")}>
          <Plus className="mr-2 h-4 w-4" />
          Add new user
        </Button>
      </CardHeader>

      <CardContent className="space-y-4">
        <div className="flex flex-wrap items-end gap-2">
          <div className="flex flex-col gap-1">
            <span className="text-sm">Email</span>
            <Input
              value={emailInput}
              onChange={(e) => setEmailInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") applyFilters();
              }}
              placeholder="Search by email"
            />
          </div>

          <div className="flex flex-col gap-1">
            <span className="text-sm">Role</span>
            <Select
              value={roleFilter}
              onValueChange={(role) => onRoleChange(role as RoleFilter)}
            >
              <SelectTrigger>
                <SelectValue placeholder={`Role: ${roleFilter}`} />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">all</SelectItem>
                <SelectItem value="user">user</SelectItem>
                <SelectItem value="editor">editor</SelectItem>
                <SelectItem value="admin">admin</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <Button type="button" onClick={applyFilters}>
            Search
          </Button>

          <div className="ml-auto text-sm">Total: {total}</div>
        </div>

        <div className="rounded-md border">
          <Table className="table-fixed w-full border-b">
            <TableHeader>
              {table.getHeaderGroups().map((hg) => (
                <TableRow key={hg.id}>
                  {hg.headers.map((h) => (
                    <TableHead key={h.id}>
                      {flexRender(h.column.columnDef.header, h.getContext())}
                    </TableHead>
                  ))}
                </TableRow>
              ))}
            </TableHeader>

            <TableBody>
              {isLoading ? (
                <TableRow>
                  <TableCell colSpan={columns.length}>Loading</TableCell>
                </TableRow>
              ) : users.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={columns.length}>No users</TableCell>
                </TableRow>
              ) : (
                table.getRowModel().rows.map((row) => (
                  <TableRow key={row.id}>
                    {row.getVisibleCells().map((cell) => (
                      <TableCell key={cell.id}>
                        {flexRender(
                          cell.column.columnDef.cell,
                          cell.getContext(),
                        )}
                      </TableCell>
                    ))}
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>

          <TablePagination
            itemsCount={total}
            page={pagination.pageIndex + 1}
            itemsPerPage={pagination.pageSize}
            onPageChange={(page) =>
              setPagination((p) => ({ ...p, pageIndex: page - 1 }))
            }
            onItemsPerPageChange={(itemsPerPage) =>
              setPagination({ pageIndex: 0, pageSize: itemsPerPage })
            }
          />
        </div>

        {isError ? (
          <div className="text-sm">
            {String((error as Error)?.message ?? "Failed to load users")}
          </div>
        ) : null}
      </CardContent>

      <UserFormDialog
        open={openDialog === "add"}
        title="Add user"
        submitLabel="Create"
        email={email}
        role={role}
        onEmailChange={setEmail}
        onRoleChange={setRole}
        onOpenChange={(open) => setOpenDialog(open ? "add" : null)}
        onSubmit={(payload) => {
          addUser.mutate(payload);
        }}
      />

      <UserFormDialog
        open={openDialog === "edit"}
        title="Edit user"
        submitLabel="Save"
        disableEmail
        email={selectedUser?.email ?? ""}
        role={role}
        onEmailChange={() => {}}
        onRoleChange={setRole}
        onOpenChange={(open) => {
          setOpenDialog(open ? "edit" : null);
          if (!open) setSelectedUser(null);
        }}
        onSubmit={(payload) => {
          if (!selectedUser) return;
          editUser.mutate({
            userId: selectedUser.id,
            user: { id: selectedUser.id, ...payload },
          });
        }}
      />

      <ConfirmDeleteDialog
        open={openDialog === "delete"}
        email={selectedUser?.email ?? ""}
        onOpenChange={(open) => {
          setOpenDialog(open ? "delete" : null);
          if (!open) setSelectedUser(null);
        }}
        onConfirm={() => {
          if (!selectedUser) return;
          deleteUser.mutate(String(selectedUser.id));
        }}
      />
    </Card>
  );
}
