FROM mcr.microsoft.com/dotnet/core/runtime:3.1 AS base
WORKDIR /app

FROM mcr.microsoft.com/dotnet/core/sdk:3.1 AS build
WORKDIR /src
COPY ["f_py_linear/f_py_linear.fsproj", "f_py_linear/"]
RUN dotnet restore "f_py_linear/f_py_linear.fsproj"
COPY . .
WORKDIR "/src/f_py_linear"
RUN dotnet build "f_py_linear.fsproj" -c Release -o /app/build

FROM build AS publish
RUN dotnet publish "f_py_linear.fsproj" -c Release -o /app/publish

FROM base AS final
WORKDIR /app
COPY --from=publish /app/publish .
ENTRYPOINT ["dotnet", "f_py_linear.dll"]
