"use client"

import { useState, useCallback, useRef, useEffect } from "react"
import { Cloud, CloudRain, Search, MapPin, Calendar, Droplets, Map, Navigation, Maximize2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { InteractiveMap } from "@/components/interactive-map"
import { ThemeToggle } from "@/components/theme-toggle"
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog"

interface SearchResult {
  display_name: string
  lat: string
  lon: string
}

interface Prediction {
  willRain: boolean
  probability: number
  amount: number
}

export default function RainPredictionPage() {
  const [latitude, setLatitude] = useState(39.8283)
  const [longitude, setLongitude] = useState(-98.5795)
  const [date, setDate] = useState(() => {
    const tomorrow = new Date()
    tomorrow.setDate(tomorrow.getDate() + 1)
    return tomorrow.toISOString().split("T")[0]
  })
  const [searchQuery, setSearchQuery] = useState("")
  const [searchResults, setSearchResults] = useState<SearchResult[]>([])
  const [isSearching, setIsSearching] = useState(false)
  const [locationName, setLocationName] = useState("")
  const [prediction, setPrediction] = useState<Prediction | null>(null)
  const [isPredicting, setIsPredicting] = useState(false)
  const [isFetchingLocation, setIsFetchingLocation] = useState(false)
  const [isMapDialogOpen, setIsMapDialogOpen] = useState(false)

  const fetchTimeoutRef = useRef<NodeJS.Timeout | null>(null)

  const fetchLocationName = useCallback(async (lat: number, lng: number) => {
    if (fetchTimeoutRef.current) {
      clearTimeout(fetchTimeoutRef.current)
    }

    fetchTimeoutRef.current = setTimeout(async () => {
      setIsFetchingLocation(true)
      try {
        const controller = new AbortController()
        const timeoutId = setTimeout(() => controller.abort(), 5000)

        const response = await fetch(
          `https://nominatim.openstreetmap.org/reverse?format=json&lat=${lat}&lon=${lng}&zoom=10&addressdetails=1`,
          {
            headers: {
              "User-Agent": "ProbiliSky Weather App",
            },
            signal: controller.signal,
          },
        )

        clearTimeout(timeoutId)

        if (response.ok) {
          const data = await response.json()
          setLocationName(data.display_name || "")
        } else {
          setLocationName("")
        }
      } catch (error) {
        if (error instanceof Error && error.name !== "AbortError") {
          setLocationName("")
        }
      } finally {
        setIsFetchingLocation(false)
      }
    }, 800)
  }, [])

  useEffect(() => {
    return () => {
      if (fetchTimeoutRef.current) {
        clearTimeout(fetchTimeoutRef.current)
      }
    }
  }, [])

  const handleLocationChange = useCallback(
    (lat: number, lng: number) => {
      setLatitude(lat)
      setLongitude(lng)
      fetchLocationName(lat, lng)
    },
    [fetchLocationName],
  )

  const handleSearch = async () => {
    if (!searchQuery.trim()) return

    setIsSearching(true)
    try {
      const response = await fetch(
        `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(searchQuery)}&countrycodes=us,ca&limit=5`,
        {
          headers: {
            "User-Agent": "ProbiliSky Weather App",
          },
        },
      )

      if (response.ok) {
        const data = await response.json()
        setSearchResults(data)
      }
    } catch (error) {
      console.error("Search error:", error)
    } finally {
      setIsSearching(false)
    }
  }

  const handleSelectResult = (result: SearchResult) => {
    const lat = Number.parseFloat(result.lat)
    const lng = Number.parseFloat(result.lon)
    const clampedLat = Math.max(25.0625, Math.min(52.9375, lat))
    const clampedLng = Math.max(-124.9375, Math.min(-67.0625, lng))
    setLatitude(clampedLat)
    setLongitude(clampedLng)
    setLocationName(result.display_name)
    setSearchResults([])
    setSearchQuery("")
  }

  const handleLatitudeChange = (value: string) => {
    const num = Number.parseFloat(value)
    if (!isNaN(num)) {
      const clamped = Math.max(25.0625, Math.min(52.9375, num))
      setLatitude(clamped)
      fetchLocationName(clamped, longitude)
    }
  }

  const handleLongitudeChange = (value: string) => {
    const num = Number.parseFloat(value)
    if (!isNaN(num)) {
      const clamped = Math.max(-124.9375, Math.min(-67.0625, num))
      setLongitude(clamped)
      fetchLocationName(latitude, clamped)
    }
  }

  const handlePredictRain = async () => {
    setIsPredicting(true)
    await new Promise((resolve) => setTimeout(resolve, 1500))

    const randomProb = Math.random() * 100
    const willRain = randomProb > 50
    const amount = willRain ? Math.round(Math.random() * 50 * 10) / 10 : 0

    setPrediction({
      willRain,
      probability: randomProb,
      amount,
    })
    setIsPredicting(false)
  }

  const formatDate = (dateStr: string) => {
    const d = new Date(dateStr)
    return d.toLocaleDateString("en-US", { year: "numeric", month: "long", day: "numeric" })
  }

  return (
    <div className="flex h-screen flex-col overflow-hidden bg-background">
      <header className="shrink-0 border-b">
        <div className="container mx-auto flex items-center justify-between px-4 py-3">
          <div className="flex items-center gap-2">
            <CloudRain className="h-7 w-7 text-primary" />
            <h1 className="text-xl font-bold">ProbiliSky</h1>
          </div>
          <ThemeToggle />
        </div>
      </header>

      <main className="flex min-h-0 flex-1 overflow-hidden">
        <div className="grid h-full w-full grid-cols-1 overflow-hidden lg:grid-cols-2">
          <div className="flex h-full flex-col gap-3 overflow-hidden border-r p-4">
            <Card className="flex min-h-0 flex-1 flex-col overflow-hidden">
              <CardHeader className="shrink-0 pb-3">
                <CardTitle className="text-lg">Location Selection</CardTitle>
                <CardDescription className="text-xs">Choose your preferred method to select a location</CardDescription>
                {locationName && (
                  <div className="mt-2 flex items-center gap-2 rounded-md border border-primary/20 bg-primary/5 px-2 py-1.5">
                    <MapPin className="h-3 w-3 shrink-0 text-primary" />
                    <div className="min-w-0 flex-1">
                      <p className="truncate text-xs font-medium">{locationName}</p>
                      <p className="font-mono text-[10px] text-muted-foreground">
                        {latitude.toFixed(4)}°N, {Math.abs(longitude).toFixed(4)}°W
                      </p>
                    </div>
                  </div>
                )}
              </CardHeader>
              <CardContent className="min-h-0 flex-1 overflow-hidden">
                <Tabs defaultValue="search" className="flex h-full flex-col">
                  <TabsList className="grid w-full shrink-0 grid-cols-3">
                    <TabsTrigger value="search" className="text-xs">
                      <Search className="mr-1 h-3 w-3" />
                      Search
                    </TabsTrigger>
                    <TabsTrigger value="map" className="text-xs">
                      <Map className="mr-1 h-3 w-3" />
                      Map
                    </TabsTrigger>
                    <TabsTrigger value="coordinates" className="text-xs">
                      <Navigation className="mr-1 h-3 w-3" />
                      Coordinates
                    </TabsTrigger>
                  </TabsList>

                  <TabsContent value="search" className="mt-3 flex-1 space-y-3 overflow-hidden">
                    <div className="space-y-3">
                      <div>
                        <Label htmlFor="search-input" className="text-sm">
                          Search for a location
                        </Label>
                        <p className="mb-2 text-xs text-muted-foreground">
                          Type an address, city, or landmark in the US or Canada
                        </p>
                        <div className="relative">
                          <div className="flex gap-2">
                            <Input
                              id="search-input"
                              placeholder="e.g., New York, NY"
                              value={searchQuery}
                              onChange={(e) => setSearchQuery(e.target.value)}
                              onKeyDown={(e) => e.key === "Enter" && handleSearch()}
                              className="text-sm"
                            />
                            <Button onClick={handleSearch} disabled={isSearching} size="sm">
                              {isSearching ? "..." : "Search"}
                            </Button>
                          </div>

                          {searchResults.length > 0 && (
                            <div className="absolute z-10 mt-1 w-full rounded-md border bg-popover shadow-lg">
                              <div className="max-h-60 overflow-y-auto p-1">
                                {searchResults.map((result, idx) => (
                                  <button
                                    key={idx}
                                    onClick={() => handleSelectResult(result)}
                                    className="w-full rounded-sm p-2 text-left text-xs transition-colors hover:bg-accent"
                                  >
                                    <div className="flex items-start gap-2">
                                      <MapPin className="mt-0.5 h-3 w-3 shrink-0 text-muted-foreground" />
                                      <span className="line-clamp-2">{result.display_name}</span>
                                    </div>
                                  </button>
                                ))}
                              </div>
                            </div>
                          )}
                        </div>
                      </div>
                      {locationName && (
                        <div className="rounded-md bg-muted p-2">
                          <p className="text-xs font-medium">Selected Location:</p>
                          <p className="text-xs text-muted-foreground">{locationName}</p>
                          <p className="text-xs text-muted-foreground">
                            {latitude.toFixed(4)}°N, {Math.abs(longitude).toFixed(4)}°W
                          </p>
                        </div>
                      )}
                    </div>
                  </TabsContent>

                  <TabsContent value="map" className="mt-3 flex-1 space-y-2 overflow-hidden">
                    <div className="flex h-full flex-col space-y-2">
                      <div className="flex shrink-0 items-center justify-between">
                        <Label className="text-sm">Click on the map to select a location</Label>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => setIsMapDialogOpen(true)}
                          className="h-7 gap-1 text-xs"
                        >
                          <Maximize2 className="h-3 w-3" />
                          Expand
                        </Button>
                      </div>
                      <div className="min-h-0 flex-1">
                        <InteractiveMap
                          latitude={latitude}
                          longitude={longitude}
                          onLocationChange={handleLocationChange}
                        />
                      </div>
                      {(latitude !== 0 || longitude !== 0) && (
                        <div className="shrink-0 rounded-md border bg-muted/50 p-4">
                          <p className="text-sm font-medium text-muted-foreground">Selected Coordinates:</p>
                          <p className="font-mono text-sm">
                            {latitude.toFixed(4)}°N, {Math.abs(longitude).toFixed(4)}°W
                          </p>
                          {isFetchingLocation && (
                            <p className="mt-1 text-xs text-muted-foreground">Loading location...</p>
                          )}
                          {locationName && !isFetchingLocation && (
                            <p className="mt-1 text-xs text-muted-foreground">{locationName}</p>
                          )}
                        </div>
                      )}
                    </div>
                  </TabsContent>

                  <TabsContent value="coordinates" className="mt-3 space-y-3 overflow-hidden">
                    <div className="space-y-2">
                      <Label htmlFor="latitude" className="text-sm font-medium">
                        Latitude
                      </Label>
                      <p className="text-xs text-muted-foreground">North-South (25.06° to 52.94°N)</p>
                      <Input
                        id="latitude"
                        type="number"
                        step="0.0001"
                        min={25.0625}
                        max={52.9375}
                        value={latitude}
                        onChange={(e) => handleLatitudeChange(e.target.value)}
                        className="text-sm"
                      />
                      <input
                        type="range"
                        min={25.0625}
                        max={52.9375}
                        step={0.0001}
                        value={latitude}
                        onChange={(e) => handleLatitudeChange(e.target.value)}
                        className="styled-slider w-full"
                        aria-label="Latitude slider"
                      />
                    </div>

                    <div className="space-y-2">
                      <Label htmlFor="longitude" className="text-sm font-medium">
                        Longitude
                      </Label>
                      <p className="text-xs text-muted-foreground">East-West (124.94°W to 67.06°W)</p>
                      <Input
                        id="longitude"
                        type="number"
                        step="0.0001"
                        min={-124.9375}
                        max={-67.0625}
                        value={longitude}
                        onChange={(e) => handleLongitudeChange(e.target.value)}
                        className="text-sm"
                      />
                      <input
                        type="range"
                        min={-124.9375}
                        max={-67.0625}
                        step={0.0001}
                        value={longitude}
                        onChange={(e) => handleLongitudeChange(e.target.value)}
                        className="styled-slider w-full"
                        aria-label="Longitude slider"
                      />
                    </div>

                    {(latitude !== 0 || longitude !== 0) && (
                      <div className="rounded-md border bg-muted/50 p-2">
                        <p className="text-sm font-medium text-muted-foreground">Location:</p>
                        {isFetchingLocation && <p className="text-sm text-muted-foreground">Loading location...</p>}
                        {locationName && !isFetchingLocation && <p className="text-sm">{locationName}</p>}
                      </div>
                    )}
                  </TabsContent>
                </Tabs>
              </CardContent>
            </Card>

            <Card className="shrink-0">
              <CardHeader className="pb-3">
                <CardTitle className="flex items-center gap-2 text-lg">
                  <Calendar className="h-4 w-4" />
                  Prediction Date
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <Input
                  type="date"
                  value={date}
                  onChange={(e) => setDate(e.target.value)}
                  min={new Date().toISOString().split("T")[0]}
                  className="text-sm"
                />

                <Button size="lg" className="w-full" onClick={handlePredictRain} disabled={isPredicting}>
                  {isPredicting ? (
                    <>
                      <Cloud className="mr-2 h-5 w-5 animate-pulse" />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <CloudRain className="mr-2 h-5 w-5" />
                      Predict Rain
                    </>
                  )}
                </Button>
              </CardContent>
            </Card>
          </div>

          <div className="flex h-full flex-col overflow-hidden p-4">
            <Card className="flex h-full flex-col overflow-hidden">
              <CardHeader className="shrink-0 pb-3">
                <CardTitle className="text-lg">Prediction Results</CardTitle>
                <CardDescription className="text-xs">Rain forecast for your selected location and date</CardDescription>
              </CardHeader>
              <CardContent className="min-h-0 flex-1 overflow-hidden">
                {prediction ? (
                  <div className="flex h-full flex-col gap-4 overflow-hidden">
                    <div
                      className={`shrink-0 rounded-lg border-2 p-4 ${
                        prediction.willRain
                          ? "border-blue-500/30 bg-blue-500/10"
                          : "border-muted-foreground/20 bg-muted/30"
                      }`}
                    >
                      <div className="mb-3 flex items-center gap-3">
                        {prediction.willRain ? (
                          <div className="rounded-full bg-blue-500/20 p-2">
                            <CloudRain className="h-6 w-6 text-blue-600 dark:text-blue-400" />
                          </div>
                        ) : (
                          <div className="rounded-full bg-muted p-2">
                            <Cloud className="h-6 w-6 text-muted-foreground" />
                          </div>
                        )}
                        <div>
                          <h3 className="text-xl font-bold">
                            {prediction.willRain ? "Rain Expected" : "No Rain Expected"}
                          </h3>
                          <p className="text-xs text-muted-foreground">{formatDate(date)}</p>
                        </div>
                      </div>
                      <div className="rounded-md bg-background/50 p-3">
                        <p className="text-xs leading-relaxed">
                          {prediction.willRain ? (
                            <>
                              There is a{" "}
                              <span className="font-bold text-blue-600 dark:text-blue-400">
                                {prediction.probability.toFixed(1)}%
                              </span>{" "}
                              chance of rain at <span className="font-semibold">{locationName || "this location"}</span>
                              . Expected rainfall is approximately{" "}
                              <span className="font-bold text-blue-600 dark:text-blue-400">{prediction.amount} mm</span>
                              .
                            </>
                          ) : (
                            <>
                              Only a <span className="font-bold">{prediction.probability.toFixed(1)}%</span> chance of
                              rain at <span className="font-semibold">{locationName || "this location"}</span>. Clear
                              skies expected.
                            </>
                          )}
                        </p>
                      </div>
                    </div>

                    <div className="grid shrink-0 gap-3 sm:grid-cols-2">
                      <div className="rounded-lg border bg-card p-4 shadow-sm">
                        <div className="mb-2 flex items-center gap-2">
                          <div className="rounded-full bg-primary/10 p-1.5">
                            <Cloud className="h-4 w-4 text-primary" />
                          </div>
                          <span className="text-xs font-medium text-muted-foreground">Probability</span>
                        </div>
                        <div className="text-2xl font-bold">{prediction.probability.toFixed(1)}%</div>
                        <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-muted">
                          <div
                            className="h-full bg-primary transition-all"
                            style={{ width: `${prediction.probability}%` }}
                          />
                        </div>
                        <p className="mt-1 text-xs text-muted-foreground">Likelihood of precipitation</p>
                      </div>

                      <div className="rounded-lg border bg-card p-4 shadow-sm">
                        <div className="mb-2 flex items-center gap-2">
                          <div className="rounded-full bg-blue-500/10 p-1.5">
                            <Droplets className="h-4 w-4 text-blue-600 dark:text-blue-400" />
                          </div>
                          <span className="text-xs font-medium text-muted-foreground">Rainfall</span>
                        </div>
                        <div className="text-2xl font-bold">{prediction.amount} mm</div>
                        <div className="mt-2 flex items-end gap-1">
                          {[...Array(5)].map((_, i) => (
                            <div
                              key={i}
                              className={`w-full rounded-t ${i < prediction.amount / 10 ? "bg-blue-500" : "bg-muted"}`}
                              style={{ height: `${Math.min(24, ((prediction.amount / 50) * 24 * (i + 1)) / 5)}px` }}
                            />
                          ))}
                        </div>
                        <p className="mt-1 text-xs text-muted-foreground">Expected accumulation</p>
                      </div>
                    </div>

                    <div className="shrink-0 rounded-lg border bg-muted/30 p-3">
                      <div className="mb-1 flex items-center gap-2 text-xs font-medium text-muted-foreground">
                        <MapPin className="h-3 w-3" />
                        Location Details
                      </div>
                      <p className="text-xs">{locationName || "Unknown location"}</p>
                      <p className="mt-0.5 font-mono text-xs text-muted-foreground">
                        {latitude.toFixed(4)}°N, {Math.abs(longitude).toFixed(4)}°W
                      </p>
                    </div>
                  </div>
                ) : (
                  <div className="flex h-full items-center justify-center">
                    <div className="text-center">
                      <div className="mx-auto mb-4 rounded-full bg-muted p-6">
                        <Cloud className="h-12 w-12 text-muted-foreground/50" />
                      </div>
                      <p className="mb-2 text-base font-semibold">No prediction yet</p>
                      <p className="text-xs text-muted-foreground">
                        Select a location and date, then click "Predict Rain"
                      </p>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </div>
      </main>

      <Dialog open={isMapDialogOpen} onOpenChange={setIsMapDialogOpen}>
        <DialogContent className="h-[98vh] max-w-[98vw] p-6">
          <DialogHeader>
            <DialogTitle>Select Location on Map</DialogTitle>
          </DialogHeader>
          <div className="flex h-[calc(100%-4rem)] flex-col gap-4">
            <div className="min-h-0 flex-1">
              <InteractiveMap
                latitude={latitude}
                longitude={longitude}
                onLocationChange={handleLocationChange}
                expandedMode={true}
              />
            </div>

            <div className="grid shrink-0 gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <Label htmlFor="dialog-latitude" className="text-sm">
                  Latitude
                </Label>
                <Input
                  id="dialog-latitude"
                  type="number"
                  step="0.0001"
                  min={25.0625}
                  max={52.9375}
                  value={latitude}
                  onChange={(e) => handleLatitudeChange(e.target.value)}
                  className="text-sm"
                />
                <input
                  type="range"
                  min={25.0625}
                  max={52.9375}
                  step={0.0001}
                  value={latitude}
                  onChange={(e) => handleLatitudeChange(e.target.value)}
                  className="styled-slider w-full"
                  aria-label="Latitude slider"
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="dialog-longitude" className="text-sm">
                  Longitude
                </Label>
                <Input
                  id="dialog-longitude"
                  type="number"
                  step="0.0001"
                  min={-124.9375}
                  max={-67.0625}
                  value={longitude}
                  onChange={(e) => handleLongitudeChange(e.target.value)}
                  className="text-sm"
                />
                <input
                  type="range"
                  min={-124.9375}
                  max={-67.0625}
                  step={0.0001}
                  value={longitude}
                  onChange={(e) => handleLongitudeChange(e.target.value)}
                  className="styled-slider w-full"
                  aria-label="Longitude slider"
                />
              </div>
            </div>

            {(latitude !== 0 || longitude !== 0) && (
              <div className="shrink-0 rounded-md border bg-muted/50 p-3">
                <p className="text-sm font-medium">
                  {latitude.toFixed(4)}°N, {Math.abs(longitude).toFixed(4)}°W
                </p>
                {isFetchingLocation && <p className="text-xs text-muted-foreground">Loading location...</p>}
                {locationName && !isFetchingLocation && <p className="text-xs text-muted-foreground">{locationName}</p>}
              </div>
            )}

            <Button onClick={() => setIsMapDialogOpen(false)} size="lg" className="w-full shrink-0">
              Confirm Location
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  )
}
