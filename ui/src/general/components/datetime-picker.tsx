import { Button } from "@/general/components/button";
import { Calendar } from "@/general/components/calendar";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/general/components/popover";
import { ScrollArea } from "@/general/components/scroll-area";
import { cn } from "@/lib/utils";
import { CalendarIcon, TrashIcon } from "lucide-react";
import { useState } from "react";

const hours = Array.from({ length: 24 }, (_, i) => i);
const minutes = Array.from({ length: 60 }, (_, i) => i);
const seconds = Array.from({ length: 60 }, (_, i) => i);

type DatetimePickerProps = {
  onChange: (date?: Date) => void;
  date?: Date;
};

/**
 * Creates a placeholder time format string based on the user's locale.
 *
 * @returns time format for current locale
 */
function getTimeFormatString() {
  const parts = new Intl.DateTimeFormat(undefined, {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).formatToParts(new Date());
  const format = parts
    .map((part) => {
      if (part.type === "day") return "DD";
      if (part.type === "month") return "MM";
      if (part.type === "year") return "YYYY";
      return part.value;
    })
    .join("");
  return `${format} hh:mm:ss`;
}

export function DatetimePicker({ onChange, date }: DatetimePickerProps) {
  const [isOpen, setIsOpen] = useState(false);

  const handleDateSelect = (selectedDate?: Date) => {
    if (!selectedDate) return;
    const newDate = date ? new Date(date) : new Date();
    newDate.setFullYear(
      selectedDate.getFullYear(),
      selectedDate.getMonth(),
      selectedDate.getDate(),
    );
    onChange(newDate);
  };

  const handleTimeChange = (
    unit: "hour" | "minute" | "second",
    value: string,
  ) => {
    if (!date) return;
    const newDate = new Date(date);
    if (unit === "hour") {
      newDate.setHours(parseInt(value, 10));
    } else if (unit === "minute") {
      newDate.setMinutes(parseInt(value, 10));
    } else if (unit === "second") {
      newDate.setSeconds(parseInt(value, 10));
    }
    onChange(newDate);
  };

  const onClear = (event: React.MouseEvent) => {
    event.stopPropagation();
    onChange();
  };

  return (
    <Popover open={isOpen} onOpenChange={setIsOpen}>
      <PopoverTrigger asChild>
        <Button
          variant="outline"
          className={cn(
            "w-full justify-start text-left font-normal",
            !date && "text-muted-foreground",
          )}
        >
          <CalendarIcon />
          {date ? date.toLocaleString() : <span>{getTimeFormatString()}</span>}
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-auto p-0">
        <div className="flex h-72">
          <div className="flex flex-col">
            <Calendar
              mode="single"
              selected={date}
              onSelect={handleDateSelect}
            />
            <Button
              variant="ghost"
              size="sm"
              className="rounded-none rounded-b-md border-t border-t-gray-200 text-destructive text-md"
              onClick={onClear}
            >
              <TrashIcon />
              Clear date & time
            </Button>
          </div>
          <div className="flex divide-x h-full">
            <ScrollArea className="w-64 sm:w-auto">
              <div className="flex sm:flex-col p-2">
                {hours.map((hour) => (
                  <Button
                    key={hour}
                    size="icon"
                    variant={
                      date && date.getHours() === hour ? "default" : "ghost"
                    }
                    className="sm:w-full shrink-0 aspect-square"
                    onClick={() => handleTimeChange("hour", hour.toString())}
                  >
                    {hour}
                  </Button>
                ))}
              </div>
            </ScrollArea>
            <ScrollArea className="w-64 sm:w-auto">
              <div className="flex flex-col p-2">
                {minutes.map((minute) => (
                  <Button
                    key={minute}
                    size="icon"
                    variant={
                      date && date.getMinutes() === minute ? "default" : "ghost"
                    }
                    className="w-full shrink-0 aspect-square"
                    onClick={() =>
                      handleTimeChange("minute", minute.toString())
                    }
                  >
                    {minute.toString().padStart(2, "0")}
                  </Button>
                ))}
              </div>
            </ScrollArea>
            <ScrollArea className="w-auto">
              <div className="flex flex-col p-2">
                {seconds.map((second) => (
                  <Button
                    key={second}
                    size="icon"
                    variant={
                      date && date.getSeconds() === second ? "default" : "ghost"
                    }
                    className="w-full shrink-0 aspect-square"
                    onClick={() =>
                      handleTimeChange("second", second.toString())
                    }
                  >
                    {second.toString().padStart(2, "0")}
                  </Button>
                ))}
              </div>
            </ScrollArea>
          </div>
        </div>
      </PopoverContent>
    </Popover>
  );
}
