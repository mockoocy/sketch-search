import { roles, type UserRole } from "@/admin/schema";
import { Button } from "@/general/components/button";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/general/components/dialog";
import { Input } from "@/general/components/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/general/components/select";

type UserFormDialogProps = {
  open: boolean;
  title: string;
  submitLabel: string;
  disableEmail?: boolean;
  onOpenChange: (open: boolean) => void;
  onSubmit: (data: { email: string; role: UserRole }) => void;
  onRoleChange: (role: UserRole) => void;
  onEmailChange?: (email: string) => void;
  email: string;
  role: UserRole;
};

export function UserFormDialog({
  open,
  title,
  submitLabel,
  disableEmail,
  onOpenChange,
  onSubmit,
  email,
  role,
  onEmailChange,
  onRoleChange,
}: UserFormDialogProps) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="space-y-4">
        <DialogHeader>
          <DialogTitle>{title}</DialogTitle>
        </DialogHeader>

        <div className="space-y-2">
          <div className="text-sm">Email</div>
          <Input
            value={email}
            onChange={(event) => onEmailChange?.(event.target.value)}
            disabled={disableEmail}
            placeholder="user@example.com"
          />
        </div>

        <div className="space-y-2">
          <div className="text-sm">Role</div>
          <Select
            value={role}
            onValueChange={(value) => onRoleChange(value as UserRole)}
          >
            <SelectTrigger>
              <SelectValue placeholder={role} />
            </SelectTrigger>
            <SelectContent>
              {roles.map((role) => (
                <SelectItem key={role} value={role}>
                  {role}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <DialogFooter>
          <Button
            type="button"
            onClick={() => onSubmit({ email: email.trim(), role })}
            disabled={!email.trim()}
          >
            {submitLabel}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
